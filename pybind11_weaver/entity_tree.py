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


def _add_child(parent, child: "Entity"):
    if child.key_in_scope in parent.children:
        # since we are traversing the AST tree in a DFS way, so when we meet a child that already exists, it is either
        # a namespace or, or some redeclaration.
        if child.cursor.kind == cindex.CursorKind.CXCursor_Namespace:
            assert len(child.children) == 0  # DFS walking should guarantee this
        else:
            _logger.warning(
                f"Entity at {child.cursor.location} already exists, skip, previous one is {self.children[child.key_in_scope].cursor.location}")
    else:
        parent.children[child.key_in_scope] = child
    assert not hasattr(child, "_entity_tree_parent")
    child._entity_tree_parent = weakref.ref(parent)


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

    def _inject_explicit_template_instantiation(self, gu: gen_unit.GenUnit):
        """
        There is no template in python side, only template instance could be exported to python.

        For pybind11-weaver, template instance is just like a normal class or normal function, the main difference is
        that these instance has a more complex identifier name.

        here is what libclang will do when encounter a template instance
        1. libclang will treat explicit template instantiation and template specialization as a normal class or function
        this is fine, as it is what we want.
        2. libclang will treat implicit template instantiation (e.g, a `Foo<T>` in function parameter) as a template reference,
        that is a problem, since we do not have an "TemplateReferenceEntity".

        Of course, we can add an "TemplateReferenceEntity" to handle this, but I think this is a bad idea, because there
        will be a lot of redundant code. A lazy solution is simple: we find all implicit template instantiation, and add
        some fake code to explicitly instantiate these template instances, and let libclang parse again, so all problem
        will be solved.

        And the code here is just to do this: 1. find all implicit template instantiation 2. add some fake code to explicitly
        instantiate these template instances 3. reload the translation unit to parse newly added code.

        """
        explicit_instantiation = set()
        implicit_instantiation = dict()

        def is_valid(cursor: cindex.Cursor):
            return gu.is_cursor_in_inputs(cursor)

        def visitor(cursor, parent, unused1):
            if not is_valid(cursor):
                return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue
            with _inject_tu([cursor, parent], gu):
                if cursor.kind in [cindex.CursorKind.CXCursor_ClassDecl,
                                   cindex.CursorKind.CXCursor_StructDecl] and common.is_concreate_template(cursor):
                    key_name = common.safe_type_reference(cursor.type)
                    explicit_instantiation.add(key_name)
                    if key_name in implicit_instantiation:
                        del implicit_instantiation[key_name]
                elif cursor.kind == cindex.CursorKind.CXCursor_TemplateRef and is_valid(cursor.referenced):
                    # we can not get more info from template ref anymore, so we need to get info from parent
                    possible_types = []
                    if parent.kind in [cindex.CursorKind.CXCursor_FieldDecl, cindex.CursorKind.CXCursor_ParmDecl,
                                       cindex.CursorKind.CXCursor_CXXBaseSpecifier]:
                        possible_types = [parent.type]
                    elif parent.kind in [cindex.CursorKind.CXCursor_FunctionDecl, cindex.CursorKind.CXCursor_CXXMethod]:
                        possible_types = [parent.result_type] + [arg.type for arg in parent.get_arguments()]
                    else:
                        _logger.info(
                            f"Only template instance may used in python will get auto exported, ignored {parent.kind} at {parent.location}")

                    for t in possible_types:
                        t = common.remove_const_ref_pointer(t).get_canonical()
                        t_c = t.get_declaration()
                        if common.is_concreate_template(t_c):
                            template_c = pylibclang._C.clang_getSpecializedCursorTemplate(t_c)
                            if is_valid(template_c):
                                key_name = common.safe_type_reference(t)
                                if key_name not in explicit_instantiation:
                                    implicit_instantiation[key_name] = _get_template_struct_class(template_c)

            return pylibclang._C.CXChildVisitResult.CXChildVisit_Recurse

        pylibclang._C.clang_visitChildren(gu.tu.cursor, visitor, pylibclang._C.voidp(0))
        init_code = "\n".join([f"{prefix} {type_name};" for type_name, prefix in implicit_instantiation.items()])
        if len(implicit_instantiation) > 0:
            _logger.info(f"Implicit template instance binding added: \n {init_code}");
        gu.reload_tu(init_code)
        funktion.FunctionEntity._added_func.clear()

    def _map_from_gu(self, gu: gen_unit.GenUnit):
        self._inject_explicit_template_instantiation(gu)
        last_parent: List[entity_base.Entity] = [None]
        worklist: List[Tuple[cindex.Cursor, entity_base.Entity]] = [(gu.tu.cursor, self)]

        def visitor(child_cursor, unused0, unused1):
            child_cursor._tu = gu.tu  # keep compatible with cindex and keep tu alive
            parent = last_parent[0]
            if gu.is_cursor_in_inputs(child_cursor):
                if child_cursor.kind == cindex.CursorKind.CXCursor_UnexposedDecl:
                    worklist.append((child_cursor, parent))
                    return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue
                new_entity = create_entity(gu, child_cursor)
                if new_entity is None:
                    return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue
                _add_child(parent, new_entity)
                worklist.append((new_entity.cursor, parent[new_entity.key_in_scope]))
            return pylibclang._C.CXChildVisitResult.CXChildVisit_Continue

        while len(worklist) > 0:
            new_item = worklist.pop()
            last_parent[0] = new_item[1]
            next_cur = new_item[0]
            pylibclang._C.clang_visitChildren(next_cur, visitor, pylibclang._C.voidp(0))
