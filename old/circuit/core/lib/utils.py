from __future__ import annotations
from ..circuit import Element, Circuit, LibName
from typing import Iterator, TypeVar

Library = dict[str, "type|Library"]
"""Element class library
    - a dict tree with leaves as `Element` classes
    - name of leaves are class names in PascalCase
    - name of branches are module names in camelCase
"""

class LibManager:
    libs: Library = {}

    @classmethod
    def load_lib(cls, lib: Library):
        name = get_library_name(lib)
        cls.libs[name] = lib

    @classmethod
    def attach_lib(cls, libnode: LibName, lib: Library):
        attach_lib(cls.libs, libnode, lib)

    @classmethod
    def get_element_class(cls, libname: LibName) -> type:
        path_2_element = parse_libname(libname)
        try:
            _next = cls.libs
            for node in path_2_element:
                if isinstance(_next, dict):
                    _next = _next[node]
                else:
                    break
            if isinstance(_next, type) and issubclass(_next, Element):
                return _next
            raise ValueError(f"invalid Element class: {_next}")
        except KeyError:
            raise KeyError(f"Element class '{libname}' not found in the library")

def iter_lib_elements(lib: Library) -> Iterator[type]:
    """iterate over all Element subclasses in the library"""
    for element in lib.values():
        if isinstance(element, dict):
            yield from iter_lib_elements(element)
        else:
            assert issubclass(element, Element)
            yield element

def get_libname(element: type) -> LibName:
    """get default libname of an Element class
    stored in element.__fields__["libname"], which is a pydantic.ModelField object
    """
    assert issubclass(element, Element)
    return element.__fields__["libname"].default

def set_libname(element: type, libname: LibName):
    assert issubclass(element, Element)
    element.__fields__["libname"].default = libname
    return
    
def create_lib(name, *elements: type|Library) -> Library:
    """create a library dict from a list of Element classes

    - modify the *libname* of each Element class to include the library name

    Args:
        name (str): library name
        *elements (type): Element classes

    Returns:
        Library: library dict
    """
    lib = {}
    for element in elements:
        if isinstance(element, dict):
            lib[get_library_name(element)] = element
        elif issubclass(element, Element):
            lib[element.__name__] = element
        else:
            raise ValueError("element must be of type Element or dict")
    for element in iter_lib_elements(lib):
        set_libname(element, f"{name}.{get_libname(element)}")
    return lib

def get_library_name(lib: Library) -> str:
    for element in iter_lib_elements(lib):
        return parse_libname(get_libname(element))[0]
    raise ValueError("empty library")

def get_libnode(lib: Library, libnode: LibName) -> Library|type:
    path = parse_libname(libnode)
    _next = lib
    for node in path:
        if isinstance(_next, dict) and node in _next:
            _next = _next[node]
        else:
            raise KeyError(f"invalid libnode: {libnode}")
    return _next

def set_libnode(lib:Library, libnode: LibName, value: Library|type):
    path = parse_libname(libnode)
    _next = lib
    for node in path[:-1]:
        if isinstance(_next, dict) and node in _next:
            _next = _next[node]
        else:
            raise KeyError(f"invalid libnode: {libnode}")
    if isinstance(_next, dict):
        _next[path[-1]] = value
        return
    raise ValueError(f"{'.'.join(path[:-1])} is not a library")

def attach_lib(parent_lib: Library, libnode: LibName, lib: Library):
    name_of_lib = get_library_name(lib)
    set_libnode(parent_lib, f"{libnode}.{name_of_lib}", lib)

    for element in iter_lib_elements(lib):
        set_libname(element, f"{libnode}.{get_libname(element)}")
    return

def read_local_lib(_locals: dict) -> Library:
    """detect all Element subclasses in the current module and return a Library object

    - supposed to be used in the __init__.py file of the library

    Example:
    ```python
    lib = read_local_lib(locals())
    ```

    Args:
        name (str): library name

    Returns:
        Library: library dict
    """
    lib = {}
    for name, _type in _locals.items():
        if isinstance(_type, type) and issubclass(_type, Element):
            lib[name] = _type
    return lib

def parse_libname(libname: LibName) -> list[str]:
    """parse libname into a list of library name, module names (optional) and element class name

    if there is only one node in the libname:
    
    - if it is in PascalCase, then is an Element belonging to the core library, add "core" as the first node
    - if it is in camelCase, then is a module name

    Args:
        libname (LibName): libname of an Element class or a library

    Returns:
        list[str]: library name, module names (optional) and element class name
    """
    if "." not in libname:
        if libname[0].isupper():
            return ["core", libname]
        return [libname]
    return libname.split(".")

def parse_element(element: Element|dict) -> Element:
    if isinstance(element, Element):
        libname = element.libname
    elif isinstance(element, dict):
        libname = element["libname"]
    else:
        raise ValueError("element must be of type Element or dict")
    element_class = LibManager.get_element_class(libname)
    return element_class.parse_obj(element)

def load_element_json(json_str: str) -> Element: 
    import json
    return parse_element(json.loads(json_str))
