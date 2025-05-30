from tracker_builder import main

def test_hello_world_output():
    expected_output = "Hello, World!"
    actual_output = main.hello_world()

    assert actual_output == expected_output, f"Expected '{expected_output}', but got '{actual_output}'"

def test_true_is_true():
    assert True

