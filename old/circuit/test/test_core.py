from siluxApi.circuit.core import Element, Node, Circuit, Link

def test_element():
    element = Element(
        name="test",
        num_of_ports=2,
    )
    assert element.ports == ["0", "1"]
    return

def test_circuit():
    elements = [
        Element(name="R", num_of_ports=2),
        Element(name="C", num_of_ports=2),
        Element(name="L", num_of_ports=2),
        Element(name="V", num_of_ports=2),
    ]
    circuit = Circuit(
        elements = elements,
        links = [
            "R vdd n1",
            "L n1 gnd",
            "C vdd gnd",
            "V vdd gnd",
        ]
    )
    circuit.connect()
    return circuit


if __name__ == "__main__":
    test_element()
    test_circuit()
