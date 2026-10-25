import promptclean
import pytest


@pytest.fixture(params=["python", "cpp"])
def clean(request):
    """Run each rule test against both implementations: the reference defines the rules, C++ must follow them."""
    return promptclean.clean_py if request.param == "python" else promptclean.clean
