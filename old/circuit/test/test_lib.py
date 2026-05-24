from siluxApi.circuit.core.lib import LibManager
from siluxApi.circuit.core.lib.utils import iter_lib_elements, get_libname

def test_core_lib():
    libs = LibManager.libs
    for element in iter_lib_elements(libs):
        print(get_libname(element))

    return
    

if __name__ == "__main__":
    test_core_lib()