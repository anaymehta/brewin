from intbase import InterpreterBase, ErrorType
from env_v1 import EnvironmentManager
from interpreterv2 import Type, Value

# Store string to enum type conversions
type_dict = {'int': Type.INT, 'bool': Type.BOOL, 'string': Type.STRING}

# FuncInfo is a class that represents information about a function
# Right now, the only thing this tracks is the line number of the first executable instruction
# of the function (i.e., the line after the function prototype: func foo)
class FuncInfo:
  def __init__(self, start_ip, params):
    self.start_ip = start_ip    # line number, zero-based
    self.environment = EnvironmentManager()
    for param in params:
      vname, vtype = param.split(':')
      self.environment.set(vname, Value(type_dict[vtype]))
    
  def pass_by_value(self, values):
    if len(self.environment.environment) != len(values):
      InterpreterBase().error(ErrorType.SYNTAX_ERROR,"Too many/few arguments to function")
    for i, vname in enumerate(self.environment.environment):
      vartype = self.environment.environment[vname].type()
      value_type = values[i]
      if vartype is not value_type.value():
        InterpreterBase().error(ErrorType.TYPE_ERROR,f"Assigning {value_type.type()} value to {vartype} variable {vname}")
      self.environment.set(vname, value_type)



# FunctionManager keeps track of every function in the program, mapping the function name
# to a FuncInfo object (which has the starting line number/instruction pointer) of that function.
class FunctionManager:
  def __init__(self, tokenized_program):
    self.func_cache = {}
    self._cache_function_line_numbers(tokenized_program)

  def get_function_info(self, func_name):
    if func_name not in self.func_cache:
      return None
    return self.func_cache[func_name]

  def _cache_function_line_numbers(self, tokenized_program):
    for line_num, line in enumerate(tokenized_program):
      if line and line[0] == InterpreterBase.FUNC_DEF:
        func_name = line[1]
        func_params = line[2:]
        func_info = FuncInfo(line_num + 1, func_params)   # function starts executing on line after funcdef
        self.func_cache[func_name] = func_info
