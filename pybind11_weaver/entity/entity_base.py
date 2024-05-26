import abc
import functools
import logging
from typing import Dict, List

from pylibclang import cindex

from pybind11_weaver import gen_unit

_logger = logging.getLogger(__name__)


def _inject_docstring(code: str, cursor: cindex.Cursor, insert_mode: str):
    if not cursor.raw_comment:
        return code
    if insert_mode == "append":
        code += f',R"_pb11_weaver({cursor.raw_comment})_pb11_weaver"'
    if insert_mode == "last_arg":
        pos = code.rfind(")")
        code = code[:pos] + f',R"_pb11_weaver({cursor.raw_comment})_pb11_weaver"' + code[pos:]
    return code


class Entity(abc.ABC):
    """Entity is like an AST Node.

    Parent is responsible to manage the life cycle of its children, and children only hold a weakref to parent.

    """

    def __init__(self, gu: gen_unit.GenUnit, cursor: cindex.Cursor):
        self.gu = gu
        self.cursor = cursor
        self.children: Dict[str, Entity] = {}

    def __contains__(self, item):
        return item in self.children

    def __getitem__(self, item: str):
        return self.children[item]

    def __setitem__(self, key, value):
        self.children[key] = value

    def python_name(self):
        # todo: implement this, this name is used to access the exported C++ target in python code.
        pass

    @property
    @functools.lru_cache
    def key_in_scope(self):
        # todo: this could be replaced by reference_name, since that name is more strong unique.
        """
        Used as key to difference different entity instances in the same scope.

        If two entity has the same key, it will be treated as a redefinition error when creating the entity tree.

        Note, C++ does allow same name in same scope as long:
        1. both are declaration
        2. both are namespace
        3. function overloading

        In pybind11-weaver, to solve these 3 case, we will
        1. ignore all redeclaration in same scope
        2. merge all namespace with the same key
        3. each function will use their signature as key, so that function overloading will map to different key.

        """
        return self.cursor.displayname

    @abc.abstractmethod
    def reference_name(self) -> str:
        """
        This name will be used in the generated code to access the C++ target being exported. e.g. if the entity is a c++
        class, this name will be the fully qualified class name, like `A::C::D::MyClass`

        It is required to reference the c++ target in any place, apparently, this name will always be fully qualified,
        so it is unique in the whole project.

        """
        pass

    @abc.abstractmethod
    def get_pb11weaver_struct_name(self) -> str:
        # todo: this name could be mangled from the reference_name.
        """
        Unique name of the entity, must be able to used as a C++ identifier.
        """
        pass

    @abc.abstractmethod
    def init_default_pybind11_value(self, parent_scope_sym: str) -> str:
        """ An expression to create pybind 11 object on stack.
        
        Args:
            parent_scope_sym: the module var name this entity should bind to

        """
        pass

    @abc.abstractmethod
    def update_stmts(self, pybind11_obj_sym: str) -> str:
        """ A group of statements to update the pybind11 object created by codes in `create_pybind11_obj_expr()`

        Args:
            pybind11_obj_sym: the pybind11 object var name

        Returns:

        """
        pass

    @abc.abstractmethod
    def default_pybind11_type_str(self) -> str:
        """ Full type of the pybind11 object."""
        pass

    def extra_code(self) -> str:
        """Entity may inject extra code into the generated binding struct."""
        return ""

    def dependency(self) -> List[str]:
        """Entity may set dependency to other entity, so that the binding struct will be generated after the dependency.

        The dependency is the reference name of the entity.
        """
        return []

    def top_level_extra_code(self) -> str:
        """Entity may inject extra code into the generated binding struct."""
        return ""
