
namespace = {}
exec(
    "import numpy\n"
    "def my_function():\n"
    "    x = 20\n"
    "    return x", namespace)
exec_result = namespace['my_function']

x = exec_result()
print(x)


