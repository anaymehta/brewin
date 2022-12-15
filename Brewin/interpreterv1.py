from intbase import *
import re
class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, input=None, trace_output=False):
        super().__init__(console_output, input)   # call InterpreterBase’s constructor
        self.variables = {}
        self.indent = [0]
        self.stack_ip = []
        self.trace_output = trace_output
        self.terminate = False
        self.indent_at_func_call = []

    def contains_comment(self, token):
        if '#' not in token: # no comment
            return False
        elif '"' not in token: # no quotes, must be comment
            return True
        else:
            quote_cnt = 0
            for c in token:
                if c == '"':
                    quote_cnt = (quote_cnt + 1) % 2
                elif c == '#':
                    if quote_cnt == 0: # not within quotes
                        return True
                    else: # within string
                        return False


    def tokenize(self, program):
        commented_program = [tokens if tokens else '' for tokens in [re.findall("(?:\".*?\"|\S)+", line) for line in program]] # maintain strings
        # remove comments
        for i, line in enumerate(commented_program):
            comment_idx = None
            for j, token in enumerate(line):
                if token.startswith('#'):
                    comment_idx = j
                    break
                elif self.contains_comment(token):
                    comment_idx = j + 1
                    commented_program[i][j] = commented_program[i][j].split('#')[0]
                    break
            if comment_idx:
                commented_program[i] = line[:comment_idx]
        
        self.tokenized_program = commented_program

        # maintain indentation
        self.program_indentation = [len(line) - len(line.lstrip(' ')) for line in program]

    def reset_interpreter(self):
        self.variables = {}
        self.indent = [0]
        self.stack_ip = []
        self.terminate = False
        self.indent_at_func_call = []

    def run(self, program):
        self.tokenize(program)
        self.reset_interpreter()
        self.ip = self.findFuncMain()

        while not self.terminate and self.ip < len(self.tokenized_program):
            self.interpret()
            self.ip += 1

    def interpret(self):
        tokenized_line = self.tokenized_program[self.ip]
        if self.trace_output:
            print(f"{self.ip}: {tokenized_line}") # tracing

        if not tokenized_line:
            return
        elif tokenized_line[0] == 'funccall': self.handle_funccall()
        elif tokenized_line[0] == 'func': self.handle_func()
        elif tokenized_line[0] == 'endfunc': self.handle_endfunc()
        elif tokenized_line[0] == 'while': self.handle_while()
        elif tokenized_line[0] == 'endwhile': self.handle_endwhile()
        elif tokenized_line[0] == 'if': self.handle_if()
        elif tokenized_line[0] == 'else': self.handle_else()
        elif tokenized_line[0] == 'endif': self.handle_endif()
        elif tokenized_line[0] == 'assign': self.handle_assign()
        elif tokenized_line[0] == 'return': self.handle_return()

    def findFuncMain(self):
        for i, x in enumerate(self.tokenized_program):
            if 'main' in x:
                return i

    def findFunc(self, func_name):
        for i, x in enumerate(self.tokenized_program):
            if 'func' in x and func_name in x:
                return i
        InterpreterBase.error(self, ErrorType(2), description= f'Function {func_name} does not exist')

    def strtoint(self, s):
        if not isinstance(s, str):
            InterpreterBase.error(self, ErrorType(1), description= 'Non-String value passed to strtoint', line_num= self.ip)
        if self.is_valid_int(s): # if valid string literal
            return int(s)
        elif self.is_valid_variable(s): # else if variable passed
            return int(self.variables[s])

    def strtobool(self, s):
        if not isinstance(s, str):
            InterpreterBase.error(self, ErrorType(1), description= 'Non-String value passed to strtobool', line_num= self.ip)
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            InterpreterBase.error(self, ErrorType(1), description= 'Non-Boolean string passed to strtobool', line_num= self.ip)

    def is_valid_int(self, value):
        return value.isnumeric() or (value[0] == '-' and value[1:].isnumeric()) # accounts for negatives

    def is_valid_name(self, variable): # used for variable and function names
        if not variable[0].isalpha(): # check that variable name starts with alphabet
            InterpreterBase.error(self, ErrorType(2), description= f'Invalid variable/function name: {variable}', line_num= self.ip)
        for ch in variable: # check rest of variable for invalid characters
            if not ch.isalnum() and ch != '_':
                InterpreterBase.error(self, ErrorType(2), description= f'Invalid variablefunction name: {variable}', line_num= self.ip)
        return variable
    
    def is_valid_variable(self, v):
        return v in self.variables

    def type_conversion(self, value):
        if '"' in value:
                return value[1:-1]
        elif value == 'True' or value == 'False':
            return self.strtobool(value)
        elif self.is_valid_int(value):
            return self.strtoint(value)
        elif self.is_valid_variable(value):
            return self.variables[value]
        else:
            InterpreterBase.error(self, ErrorType(2), description= f'Variable {value} does not exist', line_num= self.ip)

    def evaluate_expression(self, expression):
        operator = expression.pop(0)
        operators = ['+', '-', '*', '/', '%', '>', '<', '>=', '<=', '==', '!=', '&', '|']

        if operator not in operators: # if not operator, must be operand
            return self.type_conversion(operator)

        operand1 = self.evaluate_expression(expression)
        operand2 = self.evaluate_expression(expression)
        if type(operand1) != type(operand2):
            InterpreterBase.error(self, ErrorType(1), description= 'Operands are not of the same data type', line_num= self.ip)
        if operator == '+':
            return operand1 + operand2
        elif operator == '-':
            return operand1 - operand2
        elif operator == '*':
            return operand1 * operand2
        elif operator == '/':
            return operand1 // operand2
        elif operator == '%':
            return operand1 % operand2
        elif operator == '>':
            return operand1 > operand2
        elif operator == '<':
            return operand1 < operand2
        elif operator == '>=':
            return operand1 >= operand2
        elif operator == '<=':
            return operand1 <= operand2
        elif operator == '==':
            return operand1 == operand2
        elif operator == '!=':
            return operand1 != operand2
        elif operator == '&':
            return operand1 and operand2
        elif operator == '|':
            return operand1 or operand2

    def handle_value(self, value):
        if len(value) == 0:
            InterpreterBase.error(self, ErrorType(3), description= 'Missing value for assignment', line_num= self.ip)
        if len(value) > 1: # value contains an expression
            value = self.evaluate_expression(value)
        else: # value contains single value
            value = self.type_conversion(value[0])
        return value

    def get_tokenized_line(self):
        if self.indent[-1] != self.program_indentation[self.ip]:
            InterpreterBase.error(self, ErrorType(3), description= f' Indentation error: expected {self.indent[-1]} in {self.indent}, received {self.program_indentation[self.ip]}', line_num= self.ip)
        return self.tokenized_program[self.ip]

    def handle_assign(self):
        tokenized_line = self.get_tokenized_line()
        variable = self.is_valid_name(tokenized_line[1])
        value = self.handle_value(tokenized_line[2:]) # can be value or expression
        self.variables[variable] = value # assign value to variable in dictionary

    def handle_func(self):
        tokenized_line = self.get_tokenized_line()
        self.is_valid_name(tokenized_line[1])
        self.add_indent()
    
    def handle_endfunc(self):
        if self.stack_ip: # if called function, return to next line
            self.indent = self.indent_at_func_call.pop()
            self.ip = self.stack_ip.pop() # ip will increment by 1 in interpreter
        else: # if not called (main), continue
            self.del_indent()
            self.terminate = True

    def handle_while(self):
        tokenized_line = self.get_tokenized_line()
        condition = tokenized_line[1:]
        if self.evaluate_expression(condition):
            self.stack_ip.append(self.ip) # remember where while loop starts
            self.add_indent()
        else:
            end_idx = self.program_indentation.index(self.indent[-1], self.ip + 1) # find end using indentation
            self.ip = end_idx

    def handle_endwhile(self):
        self.del_indent()
        self.ip = self.stack_ip.pop() - 1 # ip will increment by 1 in interpreter

    def handle_if(self):
        tokenized_line = self.get_tokenized_line()
        condition = tokenized_line[1:]
        if self.evaluate_expression(condition):
            self.add_indent()
        else:
            end_idx = self.program_indentation.index(self.program_indentation[self.ip], self.ip + 1) # find else/end using indentation
            self.ip = end_idx - 1

    def handle_else(self):
        if self.indent[-1] > self.program_indentation[self.ip]: # if statement was executed
            end_idx = self.program_indentation.index(self.program_indentation[self.ip], self.ip + 1) # find end using indentation
            self.ip = end_idx - 1
        else: # else execute else statement
            self.add_indent()

    def handle_endif(self):
        if self.indent[-1] > self.program_indentation[self.ip]: # if statement was executed
            self.del_indent()

    def call_inbuilt_funcs(self, func_name, func_params):
        if func_name == 'input' or func_name == 'print':
            output = ""
            for param in func_params:
                output += str(self.variables[param]) if self.is_valid_variable(param) else param[1:-1] if '"' in param else param
            InterpreterBase.output(self, output)
            if func_name == 'input':
                input_val = InterpreterBase.get_input(self)
                print(type(input_val))
                self.variables['result'] = input_val
        elif func_name == 'strtoint':
            self.variables['result'] = self.strtoint(func_params[0])
        else:
            return False # not an inbuilt function
        return True # inbuilt function successfully called
            

    def handle_funccall(self):
        tokenized_line = self.get_tokenized_line()
        func_name = self.is_valid_name(tokenized_line[1])

        inbuilt_func = self.call_inbuilt_funcs(func_name, tokenized_line[2:]) # if inbuilt func, execute

        if not inbuilt_func:
            self.stack_ip.append(self.ip) # remember where while loop starts
            self.indent_at_func_call.append(self.indent)
            self.indent = [0]
            func_idx = self.findFunc(func_name)
            self.ip = func_idx - 1

    def handle_return(self):
        tokenized_line = self.get_tokenized_line()
        if len(tokenized_line) > 1: # return [expression]
            value = self.handle_value(tokenized_line[1:]) # can be value or expression
            self.variables['result'] = value # assign value to variable in dictionary
        self.indent = self.indent_at_func_call.pop()
        self.ip = self.stack_ip.pop()

    def add_indent(self):
        self.indent.append(self.program_indentation[self.ip + 1])
        # print('added indent', self.indent)

    def del_indent(self):
        # print('deleting indent', self.indent)
        self.indent.pop()