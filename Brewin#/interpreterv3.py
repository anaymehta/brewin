import copy
from enum import Enum
from env_v3 import EnvironmentManager, SymbolResult
from func_v3 import FunctionManager, FuncInfo
from intbase import InterpreterBase, ErrorType
from tokenize import Tokenizer

# Enumerated type for our different language data types
class Type(Enum):
  INT = 1
  BOOL = 2
  STRING = 3
  VOID = 4
  FUNC = 5
  OBJ = 6

# Represents a value, which has a type and its value #v3 and closure
class Value:
  def __init__(self, type, value = None, closure = None):
    self.t = type
    self.v = value
    self.c = closure

  def value(self):
    return self.v

  def set(self, other):
    self.t = other.t
    self.v = other.v
    self.c = other.c

  def type(self):
    return self.t

  def closure(self):
    return self.c

  def __str__(self):
    return f"({self.v}, {self.t})"

# Main interpreter class
class Interpreter(InterpreterBase):
  def __init__(self, console_output=True, input=None, trace_output=False):
    super().__init__(console_output, input)
    self._setup_operations()  # setup all valid binary operations and the types they work on
    self._setup_default_values()  # setup the default values for each type (e.g., bool->False)
    self.trace_output = trace_output

  # run a program, provided in an array of strings, one string per line of source code
  def run(self, program):
    self.program = program
    self._compute_indentation(program)  # determine indentation of every line
    self.tokenized_program = Tokenizer.tokenize_program(program)
    self.func_manager = FunctionManager(self.tokenized_program)
    self.ip = self.func_manager.get_function_info(InterpreterBase.MAIN_FUNC).start_ip
    self.return_stack = []
    self.terminate = False
    self.env_manager = EnvironmentManager()   # used to track variables/scope

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
      case InterpreterBase.VAR_DEF: # v2 statements
        self._define_var(args)
      case InterpreterBase.LAMBDA_DEF: # v3 statements
        self._lambda()
      case InterpreterBase.ENDLAMBDA_DEF:
        self._endlambda()
      case default:
        raise Exception(f'Unknown command: {tokens[0]}')

  def _blank_line(self):
    self._advance_to_next_statement()

  def _assign(self, tokens):
   if len(tokens) < 2:
     super().error(ErrorType.SYNTAX_ERROR,"Invalid assignment statement")
   vname = tokens[0]
   value_type = self._eval_expression(tokens[1:])
   if '.' not in vname:
    existing_value_type = self._get_value(tokens[0])
    if existing_value_type.type() != value_type.type():
      super().error(ErrorType.TYPE_ERROR,
                    f"Trying to assign a variable of {existing_value_type.type()} to a value of {value_type.type()}",
                    self.ip)
   self._set_value(tokens[0], value_type)
   self._advance_to_next_statement()

  def _funccall(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing function name to call", self.ip)
    funcvar = self.env_manager.get(args[0])
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
      # object with attribute
      if '.' in args[0]: 
        funcobj = self._get_object_value(args[0])
        if funcobj.type() != Type.FUNC:
          super().error(ErrorType.TYPE_ERROR, "Issuing funccall with member variable not of type func", self.ip)
        funcname = funcobj.value()
      # function variable
      else: 
        funcname = funcvar.value() if funcvar is not None else args[0]
      # default func - does nothing
      if funcname == '':
        self._advance_to_next_statement()
        return
      self.return_stack.append(self.ip+1)
      # closures - this
      if '.' in args[0]:
        closure = {'this': self._get_object(args[0].split('.')[0])}
      # closures - lambdas
      elif funcvar is not None:
        closure = funcvar.closure()
      else:
        closure = None
      self._create_new_environment(funcname, args[1:], closure)  # Create new environment, copy args, closure into new env
      self.ip = self._find_first_instruction(funcname)

  # create a new environment for a function call
  def _create_new_environment(self, funcname, args, closure= None):
    formal_params = self.func_manager.get_function_info(funcname)
    if formal_params is None:
        super().error(ErrorType.NAME_ERROR, f"Unknown function name {funcname}", self.ip)

    if len(formal_params.params) != len(args):
      super().error(ErrorType.NAME_ERROR,f"Mismatched parameter count in call to {funcname}, {len(args)} args for {len(formal_params.params)} params", self.ip)

    tmp_mappings = {}
    for formal, actual in zip(formal_params.params,args):
      formal_name = formal[0]
      formal_typename = formal[1]
      arg = self._get_value(actual)
      if arg.type() != self.compatible_types[formal_typename]:
        super().error(ErrorType.TYPE_ERROR,f"Mismatched parameter type for {formal_name} in call to {funcname}", self.ip)
      if formal_typename in self.reference_types:
        tmp_mappings[formal_name] = arg
      else:
        tmp_mappings[formal_name] = copy.copy(arg)

    # create a new environment for the target function
    # and add our parameters to the env
    self.env_manager.push()
    self.env_manager.import_mappings(tmp_mappings, closure)

  def _endfunc(self, return_val = None):
    if not self.return_stack:  # done with main!
      self.terminate = True
    else:
      self.env_manager.pop()  # get rid of environment for the function
      if return_val:
        self._set_result(return_val)
      else:
        # return default value for type if no return value is specified. Last param of True enables
        # creation of result variable even if none exists, or is of a different type
        return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
        if return_type != InterpreterBase.VOID_DEF:
          self._set_result(self.type_to_default[return_type])
      self.ip = self.return_stack.pop()

  def _if(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid if syntax", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean if expression", self.ip)
    if value_type.value():
      self._advance_to_next_statement()
      self.env_manager.block_nest()  # we're in a nested block, so create new env for it
      return
    else:
      for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
        if tokens[0] == InterpreterBase.ELSE_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          self.env_manager.block_nest()  # we're in a nested else block, so create new env for it
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _endif(self):
    # print(self.env_manager)
    self._advance_to_next_statement()
    self.env_manager.block_unnest()

  # we would only run this if we ran the successful if block, and fell into the else at the end of the block
  # so we need to delete the old top environment
  def _else(self):
    self.env_manager.block_unnest()   # Get rid of env for block above
    for line_num in range(self.ip+1, len(self.tokenized_program)):
      tokens = self.tokenized_program[line_num]
      if not tokens:
        continue
      if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _return(self,args):
    # do we want to support returns without values?
    return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
    default_value_type = self.type_to_default[return_type]
    if default_value_type.type() == Type.VOID:
      if args:
        super().error(ErrorType.TYPE_ERROR,"Returning value from void function", self.ip)
      self._endfunc()  # no return
      return
    if not args:
      self._endfunc()  # return default value
      return

    #otherwise evaluate the expression and return its value
    value_type = self._eval_expression(args)
    if value_type.type() != default_value_type.type():
      # print(value_type.type(), default_value_type.type())
      super().error(ErrorType.TYPE_ERROR,"Non-matching return type", self.ip)
    self._endfunc(value_type)

  def _while(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing while expression", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean while expression", self.ip)
    if value_type.value() == False:
      self._exit_while()
      return

    # If true, we advance to the next statement
    self._advance_to_next_statement()
    # And create a new scope
    self.env_manager.block_nest()

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
    super().error(ErrorType.SYNTAX_ERROR,"Missing endwhile", self.ip)

  def _endwhile(self, args):
    # first delete the scope
    self.env_manager.block_unnest()
    while_indent = self.indents[self.ip]
    cur_line = self.ip - 1
    while cur_line >= 0:
      if self.tokenized_program[cur_line][0] == InterpreterBase.WHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line -= 1
    # didn't find while
    super().error(ErrorType.SYNTAX_ERROR,"Missing while", self.ip)

  def _define_var(self, args):
    if len(args) < 2:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid var definition syntax", self.ip)
    for var_name in args[1:]:
      if self.env_manager.create_new_symbol(var_name) != SymbolResult.OK:
        super().error(ErrorType.NAME_ERROR,f"Redefinition of variable {args[1]}", self.ip)
      # is the type a valid type?
      if args[0] not in self.type_to_default:
        super().error(ErrorType.TYPE_ERROR,f"Invalid type {args[0]}", self.ip)
      # Create the variable with a copy of the default value for the type
      # print(self.type_to_default[args[0]])
      self.env_manager.set(var_name, copy.deepcopy(self.type_to_default[args[0]]))

    # print(self.env_manager)
    self._advance_to_next_statement()

  def _lambda(self): # When lambda is encountered in function (not called)
    lambdaName = self.func_manager.create_lambda_name(self.ip)
    for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if tokens[0] == InterpreterBase.ENDLAMBDA_DEF and self.indents[self.ip] == self.indents[line_num]:
          self._set_result(Value(Type.FUNC, lambdaName, copy.deepcopy(self.env_manager.environment[-1][-1])))
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endlambda", self.ip)

  def _endlambda(self, return_val= None):
    self.env_manager.pop()  # get rid of environment for the function
    if return_val:
      self._set_result(return_val)
    else:
      # return default value for type if no return value is specified. Last param of True enables
      # creation of result variable even if none exists, or is of a different type
      return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
      if return_type != InterpreterBase.VOID_DEF:
        self._set_result(self.type_to_default[return_type])
    self.ip = self.return_stack.pop()

  def _print(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid print call syntax", self.ip)
    out = []
    for arg in args:
      val_type = self._get_value(arg)
      out.append(str(val_type.value()))
    super().output(''.join(out))

  def _input(self, args):
    if args:
      self._print(args)
    result = super().get_input()
    self._set_result(Value(Type.STRING, result))   # return always passed back in result

  def _strtoint(self, args):
    if len(args) != 1:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid strtoint call syntax", self.ip)
    value_type = self._get_value(args[0])
    if value_type.type() != Type.STRING:
      super().error(ErrorType.TYPE_ERROR,"Non-string passed to strtoint", self.ip)
    self._set_result(Value(Type.INT, int(value_type.value())))   # return always passed back in result

  def _advance_to_next_statement(self):
    # for now just increment IP, but later deal with loops, returns, end of functions, etc.
    self.ip += 1

  # Set up type-related data structures
  def _setup_default_values(self):
    # set up what value to return as the default value for each type
    self.type_to_default = {}
    self.type_to_default[InterpreterBase.INT_DEF] = Value(Type.INT,0)
    self.type_to_default[InterpreterBase.STRING_DEF] = Value(Type.STRING,'')
    self.type_to_default[InterpreterBase.BOOL_DEF] = Value(Type.BOOL,False)
    self.type_to_default[InterpreterBase.VOID_DEF] = Value(Type.VOID,None)
    self.type_to_default[InterpreterBase.FUNC_DEF] = Value(Type.FUNC,'')
    self.type_to_default[InterpreterBase.OBJECT_DEF] = Value(Type.OBJ,{})

    # set up what types are compatible with what other types
    self.compatible_types = {}
    self.compatible_types[InterpreterBase.INT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.STRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.BOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.REFINT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.REFSTRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.REFBOOL_DEF] = Type.BOOL
    self.reference_types = {InterpreterBase.REFINT_DEF, Interpreter.REFSTRING_DEF,
                            Interpreter.REFBOOL_DEF}
    self.compatible_types[InterpreterBase.FUNC_DEF] = Type.FUNC
    self.compatible_types[InterpreterBase.OBJECT_DEF] = Type.OBJ

    # set up names of result variables: resulti, results, resultb
    self.type_to_result = {}
    self.type_to_result[Type.INT] = 'i'
    self.type_to_result[Type.STRING] = 's'
    self.type_to_result[Type.BOOL] = 'b'
    self.type_to_result[Type.FUNC] = 'f'
    self.type_to_result[Type.OBJ] = 'o'

  # run a program, provided in an array of strings, one string per line of source code
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

  def _find_first_instruction(self, funcname):
    func_info = self.func_manager.get_function_info(funcname)
    if not func_info:
      super().error(ErrorType.NAME_ERROR,f"Unable to locate {funcname} function")

    return func_info.start_ip

  # given a token name (e.g., x, 17, True, "foo"), give us a Value object associated with it
  def _get_value(self, token):
    if not token:
      super().error(ErrorType.NAME_ERROR,f"Empty token", self.ip)
    if token[0] == '"':
      return Value(Type.STRING, token.strip('"'))
    if token.isdigit() or token[0] == '-':
      return Value(Type.INT, int(token))
    if token == InterpreterBase.TRUE_DEF or token == Interpreter.FALSE_DEF:
      return Value(Type.BOOL, token == InterpreterBase.TRUE_DEF)
    if '.' in token:
      return self._get_object_value(token)

    # look in environments for variable
    val = self.env_manager.get(token)
    if val != None:
      return val
    # look in function manager for function
    isFunc = self.func_manager.is_function(token)
    if isFunc:
      return Value(Type.FUNC, token)
    # not found
    super().error(ErrorType.NAME_ERROR,f"Unknown variable {token}", self.ip)

  def _get_object(self, objname):
    return self.env_manager.get(objname)

  def _get_object_value(self, token):
    objname, attr = token.split('.')
    obj = self.env_manager.get(objname)
    if obj != None:
      obj = obj.value()
      if attr in obj:
        return obj[attr]
      super().error(ErrorType.NAME_ERROR,f"Unknown attribute {attr} of object {objname}", self.ip)  
    super().error(ErrorType.NAME_ERROR,f"Unknown object {objname}", self.ip)

  # given a variable name and a Value object, associate the name with the value
  def _set_value(self, varname, to_value_type):
    if '.' in varname: # is Object
      self._set_object_value(varname, to_value_type)
    else:
      value_type = self.env_manager.get(varname)
      if value_type == None:
        super().error(ErrorType.NAME_ERROR,f"Assignment of unknown variable {varname}", self.ip)
      value_type.set(to_value_type)

  def _set_object_value(self, token, to_value_type):
    objname, attr = token.split('.')
    obj = self.env_manager.get(objname)
    if obj == None:
      super().error(ErrorType.NAME_ERROR,f"Unknown object {objname}", self.ip)
    elif obj.type() != Type.OBJ:
      super().error(ErrorType.TYPE_ERROR,f"{objname} is not an object", self.ip)
    obj = obj.value()
    obj[attr] = Value(to_value_type.type(), to_value_type.value()) # Set object attribute

  # bind the result[s,i,b] variable in the calling function's scope to the proper Value object
  def _set_result(self, value_type):
    # always stores result in the highest-level block scope for a function, so nested if/while blocks
    # don't each have their own version of result
    result_var = InterpreterBase.RESULT_DEF + self.type_to_result[value_type.type()]
    self.env_manager.create_new_symbol(result_var, True)  # create in top block if it doesn't exist
    self.env_manager.set(result_var, copy.copy(value_type))

  # evaluate expressions in prefix notation: + 5 * 6 x
  def _eval_expression(self, tokens):
    stack = []

    for token in reversed(tokens):
      if token in self.binary_op_list:
        v1 = stack.pop()
        v2 = stack.pop()
        if v1.type() != v2.type():
          super().error(ErrorType.TYPE_ERROR,f"Mismatching types {v1.type()} and {v2.type()}", self.ip)
        operations = self.binary_ops[v1.type()]
        if token not in operations:
          super().error(ErrorType.TYPE_ERROR,f"Operator {token} is not compatible with {v1.type()}", self.ip)
        stack.append(operations[token](v1,v2))
      elif token == '!':
        v1 = stack.pop()
        if v1.type() != Type.BOOL:
          super().error(ErrorType.TYPE_ERROR,f"Expecting boolean for ! {v1.type()}", self.ip)
        stack.append(Value(Type.BOOL, not v1.value()))
      else:
        value_type = self._get_value(token)
        stack.append(value_type)

    if len(stack) != 1:
      super().error(ErrorType.SYNTAX_ERROR,f"Invalid expression", self.ip)

    return stack[0]

def main():
  intr = Interpreter(trace_output=True)

  test_obj_var = ['func main void',
        ' var int a b',
        ' assign a 5',
        ' assign b 7',
        ' var func f',
        ' assign f sum',
        ' funccall f a b',
        ' funccall print resulti',
        ' var object o',
        ' assign o.x 7',
        ' funccall print o.x',
        'endfunc',
        'func sum a:int b:int int',
        ' var int c',
        ' assign c + a b',
        ' return c',
        'endfunc'
  ]

  test_obj_func = [
    'func foo i:int void',
    '  funccall print i    # prints i',
    'endfunc',
    '',
    'func main void',
    '  var object x	',
    '  assign x.val 43	',
    '  assign x.our_method foo      # sets x.our_method to foo()',
    '',
    '  funccall x.our_method 42     # calls foo(42)',
    '  funccall print x.val         # prints 43',
    'endfunc'
  ]

  test_obj_func_this = [
    'func foo i:int void',
    '  assign this.val i    # sets the val member of the passed-in object',
    'endfunc',
    '',
    'func main void',
    '  var object x	',
    '  assign x.our_method foo      # sets x.our_method to foo()',
    '',
    '  funccall x.our_method 42     # calls foo(42)',
    '  funccall print x.val         # prints 42',
    'endfunc'
  ]

  test_lambda = [
    'func main void',
    '',
    '  var func xyz  ',
    '  lambda y:int int  ',
    '    var int z  ',
    '    assign z + y y  ',
    '    return z  ',
    '  endlambda  ',
    '  assign xyz resultf ',
    '  funccall xyz 50 ',
    '  funccall print resulti         # prints 100',
    'endfunc'
  ]

  test_closure = [
    'func main void',
    ' var int capture_me',
    ' assign capture_me 42',
    '',
    ' lambda a:int int             # defines a lambda for int f(int) ',
    '  return + a capture_me		# captures the capture_me variable',
    ' endlambda',
    ' # resultf holds the closure created by the lambda',
    '',
    ' var func f',
    ' assign f resultf			# f now points to the closure',
    ' assign capture_me 1     # reassign capture_me',
    ' funccall f 10			# calls our lambda function!',
    ' funccall print resulti    	# prints 52',
    'endfunc'
  ]

  test_closure_2 = [
    'func main void',
    '  var int a',
    '  var object o',
    '',
    '  assign a 5',
    '  assign o.x 10',
    '',
    '  lambda void',
    '    assign a + a 1',
    '    funccall print a        # prints 6',
    '    assign o.x 20',
    '    funccall print o.x      # prints 20',
    '  endlambda',
    '',
    '  var func f',
    '  assign f resultf',
    '  funccall f',
    '',
    '  funccall print a          # prints 5',
    '  funccall print o.x        # prints 10',
    'endfunc'
  ]

  test112 = [
    'func main void',
    '  lambda a1:int func',
    '    lambda a2:int func',
    '      lambda s3:string func',
    '        var int a3',
    '        funccall strtoint s3',
    '        assign a3 resulti',
    '        lambda a4:int func',
    '          lambda a5:int func',
    '            lambda b6:bool func',
    '              var int a6',
    '              if b6',
    '                assign a6 + a4 a2',
    '              else',
    '                assign a6 + a3 a2',
    '              endif',
    '              lambda a7:int func',
    '                lambda a8:int func',
    '                  lambda a9:int func',
    '                    lambda a10:int int',
    '                      return - + a1 - a3 * + a6 a8 a4 * a5 - a2 - a10 + a9 a7',
    '                    endlambda',
    '                    return resultf',
    '                  endlambda',
    '                  return resultf',
    '                endlambda',
    '                return resultf',
    '              endlambda',
    '              return resultf',
    '            endlambda',
    '            return resultf',
    '          endlambda',
    '          return resultf',
    '        endlambda',
    '        return resultf',
    '      endlambda',
    '      return resultf',
    '    endlambda',
    '    return resultf',
    '  endlambda',
    '  ',
    '  funccall resultf 35   # L1',
    '  funccall resultf 6    # L2',
    '  funccall resultf "1"  # L3',
    '  funccall resultf 12   # L4',
    '  funccall resultf 3    # L5',
    '  funccall resultf True # L6',
    '  funccall resultf 9    # L7',
    '  funccall resultf 2    # L8',
    '  funccall resultf 4    # L9',
    '  funccall resultf -1   # L10',
    '  funccall print resulti',
    'endfunc',
    ''
  ]

  test122 = [
    '# Church encoding',
    'func succ n:func func',
    '  lambda f:func x:int int',
    '    funccall n f x',
    '    funccall f resulti',
    '    return resulti',
    '  endlambda',
    '  return resultf',
    'endfunc',
    '',
    'func getnum n:func int',
    '  lambda x:int int',
    '    return + x 1',
    '  endlambda',
    '  funccall n resultf 0',
    '  return resulti',
    'endfunc',
    '',
    'func main void',
    '  var func zero',
    '  lambda f:func x:int int',
    '    return x',
    '  endlambda',
    '  assign zero resultf',
    '  ',
    '  funccall getnum zero',
    '  funccall print resulti',
    '  ',
    '  var func one',
    '  funccall succ zero',
    '  assign one resultf',
    '  funccall getnum one',
    '  funccall print resulti',
    'endfunc'
]

  fail27 = [
    'func main void',
    '  var int x',
    '  assign x.a 5',
    'endfunc'
  ]

  fail105 = [
    'func main void',
    '  var object o',
    '  assign o.a 100',
    '  if True',
    '    assign o.a "bar"',
    '    var object o',
    '    assign o.b True',
    '    if == o.b False',
    '      funccall print o.a',
    '    else',
    '      assign o.a 10',
    '    endif',
    '    var int y',
    '    assign y - o.a 8',
    '  endif',
    '  var int z',
    '  assign z + o.a 5',
    '  funccall print z',
    'endfunc'
  ]

  intr.run(fail105)


if __name__ == '__main__':
  main()