import logging
from typing import List, Dict, Tuple
from contextlib import contextmanager
import weakref

from pylibclang import cindex
import pylibclang._C

from pybind11_weaver import gen_unit
from pybind11_weaver.entity import create_entity
from pybind11_weaver.entity import entity_base, funktion

from pybind11_weaver.utils import common

_logger = logging.getLogger(__name__)


@contextmanager
def _inject_tu(nodes, gu):
    for node in nodes:
        node._tu = gu.tu  # keep compatible with cindex and keep tu alive
    yield nodes
    for node in nodes:
        del node._tu


def _get_template_struct_class(cursor: cindex.Cursor):
    source_file = cursor.location.file
    source_line = cursor.location.line
    # seems libclang has no API to figure out whether a template is a struct or a class, so we just read the source code
    # to find the answer ourselves.
    # read the source_line-th line's content in the source file
    with open(source_file.name, 'r') as f:
        source_code = f.read()
        line_content = source_code.splitlines()[source_line - 1]
    if "struct" in line_content:
        return "extern template struct"
    else:
        return "extern template class"


def _add_child(parent, child: "Entity") -> "Entity":
    if child is None:
        return None
    if child.key_in_scope in parent.children:
        # since we are traversing the AST tree in a BFS way, so when we meet a child that already exists, it is either
        # a namespace or some redeclaration.
        if child.cursor.kind == cindex.CursorKind.CXCursor_Namespace:
            assert len(parent.children[child.key_in_scope].children) == 0  # BFS walking should guarantee this
            return parent.children[child.key_in_scope]
        else:
            _logger.warning(
                f"Entity at {child.cursor.location} already exists, skip, previous one is {parent.children[child.key_in_scope].cursor.location}")
            return None
    else:
        parent.children[child.key_in_scope] = child
        assert not hasattr(child, "_entity_tree_parent")
        child._entity_tree_parent = weakref.ref(parent)
        return parent.children[child.key_in_scope]
    return None


class EntityTree:
    """
    Entity Tree is like the root of the AST tree, it is the entry point of the whole entity tree.

    We do not mark EntityTree as a subclass of Entity, but use duck typing to treat it as an Entity, I think this makes
    more sense to me, as EntityTree should not be an Entity.
    """

    def __init__(self, gu: gen_unit.GenUnit):
        self.children: Dict[str, entity_base.Entity] = {}
        self.gu = gu
        self._map_from_gu(gu)

    def __getitem__(self, item):
        return self.children[item]

    def __contains__(self, item):
        return item in self.children

    @property
    def cursor(self):
        return self.gu.tu.cursor

    def _inject_explicit_template_instantiation(self, gu: gen_unit.GenUnit):
        """
        There is no template in python side, only template instance could be exported to python.

        For pybind11-weaver, template instance is just like a normal class or normal function, the main difference is
        that these instance has a more complex identifier name.

        here is what libclang will do when encounter a template instance
        1. libclang will treat explicit template instantiation and template specialization as a normal class or function,
        this is fine, since it is what we want, we will be happy to see this.
        2. libclang will treat implicit template instantiation (e.g, a `Foo<T> *` in function parameter) as a template reference,
        that is a problem, because we will not generate binding for it, if the API uses an implicit instantiation, the API will
        be broken at python side.

        A lazy solution here is simple: we find all implicit template instantiation, and add
        some fake code to explicitly instantiate these template instances, then let libclang parse again, all problem
        will be solved.

        """

        explicit_instantiation = set()  # key: fully qualified type name
        implicit_instantiation = dict()  # key: fully qualified type name, value: code to explicitly instantiate it

        def should_export(cursor: cindex.Cursor):
            return gu.is_cursor_in_inputs(cursor)

        def handle_explicit(cursor: cindex.Cursor):
            instance_type_name = common.safe_type_reference(cursor.type)
            explicit_instantiation.add(instance_type_name)
            if instance_type_name in implicit_instantiation:
                del implicit_instantiation[instance_type_name]

        def handle_implicit(parent):
            possible_instances = []

            # there might be multiple template instance being used
            if parent.kind in [cindex.CursorKind.CXCursor_FieldDecl, cindex.CursorKind.CXCursor_ParmDecl,
                               cindex.CursorKind.CXCursor_CXXBaseSpecifier]:
                possible_instances = [parent.type]
            elif parent.kind in [cindex.CursorKind.CXCursor_FunctionDecl, cindex.CursorKind.CXCursor_CXXMethod]:
                possible_instances = [parent.result_type] + [arg.type for arg in parent.get_arguments()]
            else:
                _logger.info(
                    f"Only template instance may used in python will get auto exported, ignored {parent.kind} at {parent.location}")

            for instance_type in possible_instances:
                instance_type = common.remove_const_ref_pointer(instance_type).get_canonical()
                type_cursor = instance_type.get_declaration()
                if not common.is_concreate_template(type_cursor):
                    return
                template_def_cursor = pylibclang._C.clang_getSpecializedCursorTemplate(type_cursor)
                instance_type_name = common.safe_type_reference(instance_type)
                if should_export(template_def_cursor) and instance_type_name not in explicit_instantiation:
                    implicit_instantiation[instance_type_name] = _get_template_struct_class(template_def_cursor)

        def visitor(cursor, parent, unused1):
            if not should_export(cursor):
                return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue
            with _inject_tu([cursor, parent], gu):
                if cursor.kind in [cindex.CursorKind.CXCursor_ClassDecl,
                                   cindex.CursorKind.CXCursor_StructDecl] and common.is_concreate_template(cursor):
                    handle_explicit(cursor)
                elif cursor.kind == cindex.CursorKind.CXCursor_TemplateRef and should_export(cursor.referenced):
                    # we can not get more info from template ref anymore, so we need to get info from parent
                    handle_implicit(parent)
            return pylibclang._C.CXChildVisitResult.CXChildVisit_Recurse

        pylibclang._C.clang_visitChildren(gu.tu.cursor, visitor, pylibclang._C.voidp(0))

        init_code = "\n".join([f"{prefix} {type_name};" for type_name, prefix in implicit_instantiation.items()])
        if len(implicit_instantiation) > 0:
            _logger.info(f"Implicit template instance binding added: \n {init_code}");
        gu.reload_tu(init_code)

    def _map_from_gu(self, gu: gen_unit.GenUnit):
        # make sure all template instance are handled
        self._inject_explicit_template_instantiation(gu)

        # just walk the AST in BFS way, and create corresponding entity
        last_parent: List[entity_base.Entity] = [None]  # just use list to make it mutable

        # logically, a worklist item is a (cursor,entity) pair, we will map all children of cursor
        # to the children of entity, but we have some special case to handle
        # 1. if the cursor is `extern "C"`, the cursor does not create a new scope,
        # so we just add all cursor's children to the parent entity's children
        # 2. if the cursor is a namespace, since we will only have one entity in entity tree, we should add curosr's children
        # to the existing namespace entity when we meet same namespace.
        worklist: List[Tuple[cindex.Cursor, entity_base.Entity]] = [(self.cursor, self)]

        def visitor(child_cursor, unused0, unused1):
            child_cursor._tu = gu.tu  # keep compatible with cindex and keep tu alive
            parent = last_parent[0]
            if not gu.is_cursor_in_inputs(child_cursor):
                return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue

            if child_cursor.kind == cindex.CursorKind.CXCursor_LinkageSpec:
                # extern C
                worklist.append((child_cursor, parent))
                return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue

            entity_to_update = _add_child(parent, create_entity(gu, child_cursor))
            if entity_to_update is not None:
                worklist.append((child_cursor, entity_to_update))
            return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue

        while len(worklist) > 0:
            new_item = worklist.pop()
            last_parent[0] = new_item[1]
            next_cur = new_item[0]
            pylibclang._C.clang_visitChildren(next_cur, visitor, pylibclang._C.voidp(0))
