from enum import Enum
from intbase import InterpreterBase, ErrorType
from env_v1 import EnvironmentManager
from tokenize import Tokenizer
from copy import deepcopy

# Enumerated type for our different language data types
class Type(Enum):
  INT = 1
  BOOL = 2
  STRING = 3
  REFINT = 4
  REFBOOL = 5
  REFSTRING = 6

# Represents a value, which has a type and its value
class Value:
  def __init__(self, type, value = None):
    self.t = type
    self.v = value

  def value(self):
    return self.v

  def set(self, other):
    self.t = other.t
    self.v = other.v

  def type(self):
    return self.t

  def __str__(self):
    return f"({self.v}, {self.t})"

# Store string to enum type conversions
TYPE_DICT = {'int': Type.INT, 'bool': Type.BOOL, 'string': Type.STRING, 'refint': Type.REFINT, 'refbool': Type.REFBOOL, 'refstring': Type.REFSTRING}
VAL_TYPES = {Type.INT, Type.BOOL, Type.STRING}
REF_TYPES = {Type.REFINT, Type.REFBOOL, Type.REFSTRING}
REF_TO_INT = 3
RETURN_TYPES = {'int', 'bool', 'string', 'void'}

# FuncInfo is a class that represents information about a function
# Right now, the only thing this tracks is the line number of the first executable instruction
# of the function (i.e., the line after the function prototype: func foo)
class FuncInfo:
  def __init__(self, start_ip, params, return_type):
    self.start_ip = start_ip    # line number, zero-based
    self.environment = EnvironmentManager()
    self.refs = set()
    for param in params:
      vname, vtype = param.split(':')
      self.environment.set(vname, Value(TYPE_DICT[vtype]))
      if TYPE_DICT[vtype] in REF_TYPES:
        self.refs.add(vname)
    if return_type not in RETURN_TYPES:
      InterpreterBase().error(ErrorType.NAME_ERROR,"Invalid return type")
    self.return_type = return_type

  def pass_params(self, args, dict):
    if len(self.environment.get_all()) < len(args):
      InterpreterBase().error(ErrorType.NAME_ERROR,"Too many arguments to function")
    elif len(self.environment.get_all()) > len(args):
      print(self.environment.get_all())
      print(args)
      InterpreterBase().error(ErrorType.NAME_ERROR,"Too few arguments to function")
    for i, vname in enumerate(self.environment.get_all()):
      vartype = self.environment.get_all()[vname][0].type()
      value_type = self._args_to_vals(args[i], dict)
      if self._type_mismatch(vartype, value_type.type()):
        InterpreterBase().error(ErrorType.TYPE_ERROR,f"Assigning {value_type.type()} value to {vartype} variable {vname}")
      if vname in self.refs:
        self.environment.set(vname, value_type, declared= 1, ref= args[i])
      else:
        self.environment.set(vname, value_type)

  # given an arg (e.g., x, 17, True, "foo"), give us a Value object associated with it  
  def _args_to_vals(self, arg, dict):
    if not arg:
      super().error(ErrorType.NAME_ERROR,f"Empty arg", self.ip) #no
    if arg[0] == '"':
      return Value(Type.STRING, arg.strip('"'))
    if arg.isdigit() or arg[0] == '-':
      return Value(Type.INT, int(arg))
    if arg == InterpreterBase.TRUE_DEF or arg == InterpreterBase.FALSE_DEF:
      return Value(Type.BOOL, arg == InterpreterBase.TRUE_DEF)
    value = dict.get(arg)[0]
    if value  == None:
      super().error(ErrorType.NAME_ERROR,f"Unknown variable {arg}", self.ip) #!
    return value
  
  def _type_mismatch(self, type1, type2):
    if type1 is not type2 and abs(type1.value - type2.value) != REF_TO_INT:
      return True
    return False

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
        func_params = line[2:-1]
        func_return_type = line[-1]
        func_info = FuncInfo(line_num + 1, func_params, func_return_type)   # function starts executing on line after funcdef
        self.func_cache[func_name] = func_info

# Main interpreter class
class Interpreter(InterpreterBase):
  def __init__(self, console_output=True, input=None, trace_output=False):
    super().__init__(console_output, input)
    self._setup_operations()  # setup all valid binary operations and the types they work on
    self.trace_output = trace_output

  # run a program, provided in an array of strings, one string per line of source code
  def run(self, program):
    self.program = program
    self._compute_indentation(program)  # determine indentation of every line
    self.tokenized_program = Tokenizer.tokenize_program(program)
    self.func_manager = FunctionManager(self.tokenized_program)
    self.env_manager = [[]]
    self.func_stack = []
    self.ip = self._find_first_instruction(InterpreterBase.MAIN_FUNC, [])
    self.return_stack = []
    self.terminate = False

    # main interpreter run loop
    while not self.terminate:
      self._process_line()

  def _process_line(self):
    if self.trace_output:
      print(f"{self.ip:04}: {self.program[self.ip].rstrip()}")
    tokens = self.tokenized_program[self.ip]
    if not tokens:
      self._blank_line()
      return

    args = tokens[1:]

    match tokens[0]:
      case InterpreterBase.ASSIGN_DEF:
        self._assign(args)
      case InterpreterBase.VAR_DEF:
        self._var(args)
      case InterpreterBase.FUNCCALL_DEF:
        self._funccall(args)
      case InterpreterBase.ENDFUNC_DEF:
        self._endfunc()
      case InterpreterBase.IF_DEF:
        self._if(args)
      case InterpreterBase.ELSE_DEF:
        self._else()
      case InterpreterBase.ENDIF_DEF:
        self._endif()
      case InterpreterBase.RETURN_DEF:
        self._return(args)
      case InterpreterBase.WHILE_DEF:
        self._while(args)
      case InterpreterBase.ENDWHILE_DEF:
        self._endwhile(args)
      case default:
        raise Exception(f'Unknown command: {tokens[0]}')

  def _blank_line(self):
    self._advance_to_next_statement()

  def _assign(self, tokens):
    if len(tokens) < 2:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid assignment statement", self.ip) #no
    vname = tokens[0]
    if not self._is_var(vname):
      super().error(ErrorType.NAME_ERROR,f"Assigning value to undeclared variable {vname}", self.ip)
    value_type = self._eval_expression(tokens[1:])
    vartype = self.env_manager[-1][-1].get(vname)[0].type()
    varref = self.env_manager[-1][-1].get(vname)[2]
    if vartype != value_type.type():
      super().error(ErrorType.TYPE_ERROR,f"Assigning {value_type.type()} value to {vartype} variable {vname}", self.ip)
    self._set_value(vname, value_type, 0, varref)
    self._advance_to_next_statement()

  def _var(self, tokens):
    if len(tokens) < 2:
     super().error(ErrorType.SYNTAX_ERROR,"Invalid variable declaration", self.ip) #no
    vtype = tokens[0]
    vnames = tokens[1:]
    for vname in vnames:
      if self._cant_shadow_var(vname):
        super().error(ErrorType.NAME_ERROR,f"Variable {vname} is already declared in this scope", self.ip)
      value_type = self._def_val(vtype)
      self._update_shadowed_var(vname)
      self._set_value(vname, value_type)
    self._advance_to_next_statement()

  def _funccall(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing function name to call", self.ip) #!
    if args[0] == InterpreterBase.PRINT_DEF:
      self._print(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.INPUT_DEF:
      self._input(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.STRTOINT_DEF:
      self._strtoint(args[1:])
      self._advance_to_next_statement()
    else:
      self.return_stack.append(self.ip+1)
      self.ip = self._find_first_instruction(args[0], args[1:])

  def _endfunc(self):
    if len(self.func_stack) > len(self.return_stack): # no return statement in func
      self._return([]) # return with no args
      return
    if not self.return_stack:  # done with main!
      self.terminate = True
    else:
      self.ip = self.return_stack.pop()
      self._copy_func_refs()

  def _if(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid if syntax", self.ip) #no
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean if expression", self.ip) #!
    if value_type.value():
      self._enter_block()
      self._advance_to_next_statement()
      return
    else:
      for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if (tokens[0] == InterpreterBase.ENDIF_DEF or tokens[0] == InterpreterBase.ELSE_DEF) and self.indents[self.ip] == self.indents[line_num]:
          if tokens[0] == InterpreterBase.ELSE_DEF:
            self._enter_block()
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip) #no

  def _endif(self):
    self._exit_block()
    self._advance_to_next_statement()

  def _else(self):
    for line_num in range(self.ip+1, len(self.tokenized_program)):
      tokens = self.tokenized_program[line_num]
      if not tokens:
        continue
      if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self._exit_block()
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip) #no

  def _return(self,args):
    return_type = self.func_manager.get_function_info(self.func_stack.pop()).return_type
    if return_type == InterpreterBase.VOID_DEF:
      if not args:
        self._endfunc()
        return
      else:
        super().error(ErrorType.TYPE_ERROR,f"Void function cannot return a value", self.ip)
    value_type = self._eval_expression(args) if args else self._def_val(return_type)
    if value_type.type() != TYPE_DICT[return_type]:
      super().error(ErrorType.TYPE_ERROR,f"Returned value {value_type.value()} does not match function return type {return_type}", self.ip)
    self._endfunc()
    self._set_result(value_type)  # return always passed back in result

  def _while(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing while expression", self.ip) #no
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean while expression", self.ip) #!
    if value_type.value() == False:
      self._exit_while()
      return

    # If true, we advance to the next statement
    self._enter_block()
    self._advance_to_next_statement()

  def _exit_while(self):
    while_indent = self.indents[self.ip]
    cur_line = self.ip + 1
    while cur_line < len(self.tokenized_program):
      if self.tokenized_program[cur_line][0] == InterpreterBase.ENDWHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line + 1
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line += 1
    # didn't find endwhile
    super().error(ErrorType.SYNTAX_ERROR,"Missing endwhile", self.ip) #no

  def _endwhile(self, args):
    while_indent = self.indents[self.ip]
    cur_line = self.ip - 1
    while cur_line >= 0:
      if self.tokenized_program[cur_line][0] == InterpreterBase.WHILE_DEF and self.indents[cur_line] == while_indent:
        self._exit_block()
        self.ip = cur_line
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line -= 1
    # didn't find while
    super().error(ErrorType.SYNTAX_ERROR,"Missing while", self.ip) #no

  def _print(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid print call syntax", self.ip) #no
    out = []
    for arg in args:
      val_type = self._get_value(arg)
      out.append(str(val_type.value()))
    super().output(''.join(out))

  def _input(self, args):
    if args:
      self._print(args)
    result = super().get_input()
    self._set_result(Value(Type.STRING, result))

  def _strtoint(self, args):
    if len(args) != 1:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid strtoint call syntax", self.ip) #no
    value_type = self._get_value(args[0])
    if value_type.type() != Type.STRING:
      super().error(ErrorType.TYPE_ERROR,"Non-string passed to strtoint", self.ip) #!
    self._set_result(Value(Type.INT, int(value_type.value())))

  def _advance_to_next_statement(self):
    # for now just increment IP, but later deal with loops, returns, end of functions, etc.
    self.ip += 1

  # create a lookup table of code to run for different operators on different types
  def _setup_operations(self):
    self.binary_op_list = ['+','-','*','/','%','==','!=', '<', '<=', '>', '>=', '&', '|']
    self.binary_ops = {}
    self.binary_ops[Type.INT] = {
     '+': lambda a,b: Value(Type.INT, a.value()+b.value()),
     '-': lambda a,b: Value(Type.INT, a.value()-b.value()),
     '*': lambda a,b: Value(Type.INT, a.value()*b.value()),
     '/': lambda a,b: Value(Type.INT, a.value()//b.value()),  # // for integer ops
     '%': lambda a,b: Value(Type.INT, a.value()%b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.STRING] = {
     '+': lambda a,b: Value(Type.STRING, a.value()+b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.BOOL] = {
     '&': lambda a,b: Value(Type.BOOL, a.value() and b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '|': lambda a,b: Value(Type.BOOL, a.value() or b.value())
    }

  def _compute_indentation(self, program):
    self.indents = [len(line) - len(line.lstrip(' ')) for line in program]

  def _find_first_instruction(self, funcname, args):
    func_info = deepcopy(self.func_manager.get_function_info(funcname))
    if func_info == None:
      super().error(ErrorType.NAME_ERROR,f"Unable to locate {funcname} function", self.ip) #!
    self.func_stack.append(funcname)
    if self.env_manager[-1]:
      func_info.pass_params(args, self.env_manager[-1][-1].get_all())
    else: # main
      func_info.pass_params(args, {})
    self.env_manager.append([func_info.environment])
    return func_info.start_ip

  # given a token name (e.g., x, 17, True, "foo"), give us a Value object associated with it
  def _get_value(self, token):
    if not token:
      super().error(ErrorType.NAME_ERROR,f"Empty token", self.ip) #no
    if token[0] == '"':
      return Value(Type.STRING, token.strip('"'))
    if token.isdigit() or token[0] == '-':
      return Value(Type.INT, int(token))
    if token == InterpreterBase.TRUE_DEF or token == InterpreterBase.FALSE_DEF:
      return Value(Type.BOOL, token == InterpreterBase.TRUE_DEF)
    value = self.env_manager[-1][-1].get(token)[0]
    if value  == None:
      super().error(ErrorType.NAME_ERROR,f"Unknown variable {token}", self.ip) #!
    return value

  # given a variable name and a Value object, associate the name with the value
  def _set_value(self, varname, value_type, declared= 1, ref= None):
    self.env_manager[-1][-1].set(varname, value_type, declared, ref)

  # evaluate expressions in prefix notation: + 5 * 6 x
  def _eval_expression(self, tokens):
    stack = []

    for token in reversed(tokens):
      if token in self.binary_op_list:
        v1 = stack.pop()
        v2 = stack.pop()
        if v1.type() != v2.type():
          super().error(ErrorType.TYPE_ERROR,f"Mismatching types {v1.type()} and {v2.type()}", self.ip) #!
        operations = self.binary_ops[v1.type()]
        if token not in operations:
          super().error(ErrorType.TYPE_ERROR,f"Operator {token} is not compatible with {v1.type()}", self.ip) #!
        stack.append(operations[token](v1,v2))
      elif token == '!':
        v1 = stack.pop()
        if v1.type() != Type.BOOL:
          super().error(ErrorType.TYPE_ERROR,f"Expecting boolean for ! {v1.type()}", self.ip) #!
        stack.append(Value(Type.BOOL, not v1.value()))
      else:
        value_type = self._get_value(token)
        stack.append(value_type)

    if len(stack) != 1:
      super().error(ErrorType.SYNTAX_ERROR,f"Invalid expression", self.ip) #no

    return stack[0]

  def _is_var(self, varname):
    return self.env_manager[-1][-1].get(varname)[0] is not None

  def _cant_shadow_var(self, varname):
    value_type, declared, ref = self.env_manager[-1][-1].get(varname)
    return value_type is not None and declared

  def _update_shadowed_var(self, varname):
    if len(self.env_manager[-1]) < 2:
      return
    outer_env = self.env_manager[-1][-2]
    outer_var = outer_env.get(varname)
    if outer_var[0] is not None:
      outer_env.set(varname, self.env_manager[-1][-1].get(varname)[0], outer_var[1], outer_var[2])

  def _def_val(self, vtype):
    match vtype:
        case InterpreterBase.INT_DEF:
          return self._get_value('0')
        case InterpreterBase.STRING_DEF:
          return self._get_value('""')
        case InterpreterBase.BOOL_DEF:
          return self._get_value('False')

  def _set_result(self, value_type): # Set to func scope
    scopes = len(self.env_manager[-1])
    for scope in range(scopes):
      match value_type.type(): # return always passed back in result
        case Type.INT:
          self.env_manager[-1][scope].set(InterpreterBase.RESULT_DEF + 'i', value_type, 1, None)
        case Type.BOOL:
          self.env_manager[-1][scope].set(InterpreterBase.RESULT_DEF + 'b', value_type, 1, None)
        case Type.STRING:
          self.env_manager[-1][scope].set(InterpreterBase.RESULT_DEF + 's', value_type, 1, None)

  def _enter_block(self):
    block_env = deepcopy(self.env_manager[-1][-1])
    block_env.set_undeclared()
    self.env_manager[-1].append(block_env)

  def _exit_block(self):
    cur_block = self.env_manager[-1][-1].get_all()
    prev_block = self.env_manager[-1][-2].get_all()
    for vars in cur_block:
      if cur_block[vars][1] == 0: # 0 = Undeclared, 1 = Declared
        prev_block[vars] = (cur_block[vars][0], prev_block[vars][1], prev_block[vars][2])
    self.env_manager[-1].pop()
  
  def _debug_blocks(self):
    blocks = len(self.env_manager[-1])
    for block in range(blocks):
      dict = self.env_manager[-1][block].get_all()
      for key in dict:
        val = dict[key]
        print(f"{key}: {val[0]}, {val[1]}, {val[2]}")
    print('\n')

  def _copy_func_refs(self):
    if len(self.env_manager) <= 1: # returning from main
      return
    cur_block = self.env_manager[-1][-1].get_all()
    prev_block = self.env_manager[-2][-1].get_all()
    for vars in cur_block:
      ref = cur_block[vars][2]
      if ref is not None:
        prev_block[ref] = (cur_block[vars][0], prev_block[ref][1], prev_block[ref][2])
    self.env_manager.pop()

def main():
  interp = Interpreter(trace_output=True)

  # Valid. Tests parameter passing, type
  p1 = ['func main void', ' var int a b', ' funccall abc 1 2', ' assign b 5', ' funccall print a b', 'endfunc', '', 'func abc x:int y:int void', ' var int c', ' assign c + x y', ' funccall print c', 'endfunc']

  # Valid. Tests parameter passing, result
  p2 = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 7',
        ' funccall sum a b',
        ' funccall print resulti',
        'endfunc',
        'func sum a:int b:int int',
        ' var int c',
        ' assign c + a b',
        ' funccall print c',
        ' return c',
        'endfunc'
  ]

  # Invalid. Incorrect return type
  p3 = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 7',
        ' funccall sum a b',
        ' funccall print resulti',
        'endfunc',
        'func sum a:int b:int void',
        ' var int c',
        ' assign c + a b',
        ' funccall print c',
        ' return c',
        'endfunc'
  ]

  # Valid. Default return value
  p4 = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 7',
        ' funccall sum a b',
        ' funccall print resulti',
        'endfunc',
        'func sum a:int b:int int',
        ' var int c',
        ' assign c + a b',
        ' funccall print c',
        # ' return',
        'endfunc'
  ]

  # Valid. if/else, input/print
  p5 = ['func main void',
        ' var int a b',
        ' funccall input "Enter word: "',
        ' funccall print results',
        ' assign a 5',
        ' assign b 7',
        ' funccall sum a b',
        ' funccall print resulti',
        'endfunc',
        'func sum a:int b:int int',
        ' var int c',
        ' assign c + a b',
        ' if > c 10',
        '  assign c * 2 c',
        ' else',
        '  assign c / c 2',
        ' endif',
        ' funccall print c',
        ' return c',
        'endfunc'
  ]

  # Valid. while loop
  p6 = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 7',
        ' funccall sum a b',
        ' funccall print resulti',
        ' while > a 3',
        '  funccall print a',
        '  assign a - a 1',
        ' endwhile',
        ' funccall print a',
        'endfunc',
        'func sum a:int b:int int',
        ' var int c',
        ' assign c + a b',
        ' if > c 10',
        '  assign c * 2 c',
        ' else',
        '  assign c / c 2',
        ' endif',
        ' funccall print c',
        ' return c',
        'endfunc'
  ]

  # Valid. Shadowing
  p7 = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 7',
        ' funccall sum a b',
        ' funccall print resulti',
        'endfunc',
        'func sum a:int b:int int',
        ' funccall print a',
        ' var int c',
        ' assign c + a b',
        ' if > c 10',
        '  var bool a',
        '  assign a False',
        '  assign c * 2 100',
        '  funccall print a',
        ' else',
        '  assign c / c 2',
        ' endif',
        ' funccall print a',
        ' funccall print c',
        ' return c',
        'endfunc'
  ]

  # Valid. Refint
  p8 = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 7',
        ' funccall sum a b',
        ' funccall print resulti',
        ' funccall print b',
        'endfunc',
        'func sum a:int b:refint int',
        ' funccall print a',
        ' var int c',
        ' assign c + a b',
        ' if > c 10',
        '  var int a',
        '  assign a 2',
        '  funccall sum b a',
        '  assign c * 2 a',
        '  funccall print resulti',
        ' else',
        '  assign c / c 2',
        ' endif',
        ' assign b 99',
        ' funccall print b',
        ' return c',
        'endfunc'
  ]

  # Valid
  t2 = [
    'func foo a:int b:int c:int void',
    '  assign a 1',
    '  assign b 1',
    '  assign c 1',
    '  funccall print a " " b " " c',
    'endfunc',
    '',
    '',
    'func main void',
    '  var int a b c',
    '  assign a 2',
    '  assign b 2',
    '  assign c 2',
    '  funccall print a " " b " " c',
    '  funccall foo a b c',
    '  funccall print a " " b " " c',
    'endfunc'
  ]

  # Valid. Assigns val to var, then shadows it
  t3 = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 100',
        ' funccall print a " " b',
        ' if True',
        '   assign a 10',
        '   funccall print a " " b',
        '   var int a',
        '   assign a 6',
        '   funccall print a " " b',
        '   var int b',
        '   assign b -1',
        '   funccall print a " " b',
        ' endif',
        ' funccall print a " " b',
        'endfunc'
]

  # Valid. Pass by ref
  t13 = [
    'func main void',
    '  var int x',
    '  var string y',
    '  var bool z',
    '  assign x 42',
    '  assign y "foo"',
    '  assign z True ',
    '  funccall foo x y z',
    '  funccall print x " " y " " z',
    '  funccall bletch',
    '  funccall bar resulti',
    '  funccall print resulti',
    'endfunc',
    '',
    'func foo a:refint b:refstring c:refbool void',
    ' assign a -42',
    ' assign b "bar"',
    ' assign c False',
    'endfunc',
    '',
    'func bletch int',
    ' return 100',
    'endfunc',
    '',
    'func bar a:refint void',
    ' assign a -100 ',
    'endfunc',
  ]

  t9 = [
    'func foo void',
    ' return 5',
    'endfunc',
    '',
    'func main void',
    ' funccall foo',
    'endfunc',
    ''
  ]

  # Recursion
  t47 = [
    '#double a string, recursively',
    '',
    'func main void',
    '    var int n',
    '    assign n 4',
    '    var string result',
    '    assign result "a"',
    '    funccall double result n',
    '    funccall print result',
    '',
    '    assign n 6',
    '    assign result "##"',
    '    funccall double result n',
    '    funccall print result',
    '',
    'endfunc',
    '',
    'func double result:refstring n:int void',
    '    if == n 0',
    '        return',
    '    endif',
    '    assign n - n 1',
    '    assign result + result result',
    '    funccall double result n',
    'endfunc',
    ''
  ]

  t99 = [
    '#double a string, recursively',
    '',
    'func main void',
    '    var int n',
    '    assign n 4',
    '    var string result',
    '    assign result "a"',
    '    funccall double result n',
    '    funccall print result',
    '',
    '    assign n 6',
    '    assign result "##"',
    '    funccall double result n',
    '    funccall print result',
    '    funccall print n',
    '',
    'endfunc',
    '',
    'func double result:refstring n:int void',
    '    if == n 0',
    '        var int n',
    '        funccall print n',
    '        return',
    '    endif',
    '    assign n - n 1',
    '    assign result + result result',
    '    funccall double result n',
    'endfunc',
    ''
  ]

  t100 = [
    '# Equivalent to: bool absval(int val, int& change_me)',
'func absval val:int change_me:refint bool',
'  if < val 0',
'    assign change_me * -1 val',
'    return True',
'  else',
'    assign change_me val',
'    return False',
'  endif',
'endfunc',
'',
'func main void',
'  var int val output',
'  assign val -5',
'  funccall absval val output',
'  funccall print "The absolute value is: " output',
'  funccall print "Did I negate the input value? " resultb',
'endfunc',
''
  ]

  interp.run(p8)

if __name__ == '__main__':
  main()