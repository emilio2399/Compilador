#Emilio Castillo Alegría - A01631303

DIGITS = '0123456789'

t_int				= 'INT'
t_float  			= 'FLOAT'
t_plus   		 	= 'PLUS'
t_minus    			= 'MINUS'
t_multplication     = 'MUL'
t_division    		= 'DIV'
t_lparen   			= 'LPAREN'
t_rparen   			= 'RPAREN'
t_eof				= 'EOF'

class Error:
	def __init__(self, pos_start, pos_end, error_name, details):
		self.pos_start = pos_start
		self.pos_end = pos_end
		self.error_name = error_name
		self.details = details
	
	def as_string(self):
		result  = f'{self.error_name}: {self.details}\n'
		result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
		return result

class IllegalCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
	def __init__(self, pos_start, pos_end, details=''):
		super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
	def __init__(self, pos_start, pos_end, details, context):
		super().__init__(pos_start, pos_end, 'Runtime Error', details)
		self.context = context

class Position:
	def __init__(self, idx, ln, col, fn, ftxt):
		self.idx = idx
		self.ln = ln
		self.col = col
		self.fn = fn
		self.ftxt = ftxt

	def advance(self, current_pos=None):
		self.idx += 1
		self.col += 1

		if current_pos == '\n':
			self.ln += 1
			self.col = 0

		return self

	def copy(self):
		return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

class Token:
	def __init__(self, type_, value=None, pos_start=None, pos_end=None):
		self.type = type_
		self.value = value

		if pos_start:
			self.pos_start = pos_start.copy()
			self.pos_end = pos_start.copy()
			self.pos_end.advance()

		if pos_end:
			self.pos_end = pos_end.copy()

	def matches(self, type_, value):
		return self.type == type_ and self.value == value
	
	def __repr__(self):
		if self.value: return f'{self.type}:{self.value}'
		return f'{self.type}'





class Lexer:
	def __init__(self, fn, text):
		self.text = text
		self.pos = -1
		self.current_loc = None
		self.advance()

	def advance(self):
		self.pos +=1
		self.current_loc = self.text[self.pos] if self.pos < len(self.text) else None

	def make_tokens(self):
		tokens = []

		while self.current_loc != None:
			if self.current_loc in ' \t':
				self.advance()
			elif self.current_loc == '+':
				tokens.append(Token(t_plus))
				self.advance()
			elif self.current_loc == '-':
				tokens.append(Token(t_minus))
				self.advance()
			elif self.current_loc == '*':
				tokens.append(Token(t_multplication))
				self.advance()
			elif self.current_loc == '/':
				tokens.append(Token(t_division))
				self.advance()
			elif self.current_loc == '(':
				tokens.append(Token(t_lparen))
				self.advance()
			elif self.current_loc == ')':
				tokens.append(Token(t_rparen))
				self.advance()
			else:
				char = self.current_loc
				self.advance()
				return [], IllegalCharError("'", char, "'" + char + "'")

		tokens.append(Token(t_eof, pos_start=self.pos))
		return tokens, None


	def number_creation(self):
		num_str = ''
		dot_count = 0
		pos_start = self.pos.copy()

		while self.current_loc != None and self.current_loc in DIGITS + '.':
			if self.current_loc == '.':
				if dot_count == 1: break
				dot_count += 1
			num_str += self.current_loc
			self.advance()


		if dot_count == 0:
			return Token(t_int, int(num_str), pos_start, self.pos)
		else:
			return Token(t_float, float(num_str), pos_start, self.pos)


	
class NumberNode:
	def __init__(self, tok):
		self.tok = tok

		self.pos_start = self.tok.pos_start
		self.pos_end = self.tok.pos_end

	def __repr__(self):
		return f'{self.tok}'


class BinOpNode:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

	def __repr__(self):
		return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class UnaryOpNode:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

		self.pos_start = self.op_tok.pos_start
		self.pos_end = node.pos_end

	def __repr__(self):
		return f'({self.op_tok}, {self.node})'


class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.last_registered_advance_count = 0
		self.advance_count = 0

	def register_advancement(self):
		self.last_registered_advance_count = 1
		self.advance_count += 1

	def register(self, res):
		self.last_registered_advance_count = res.advance_count
		self.advance_count += res.advance_count
		if res.error: self.error = res.error
		return res.node

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		if not self.error or self.last_registered_advance_count == 0:
			self.error = error
		return self

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tok_idx = -1
		self.advance()

	def advance(self, ):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok

	def parse(self):
		res = self.expr()
		if not res.error and self.current_tok.type != t_EOF:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected '+', '-', '*', '/'"
			))
		return res
   

	def factor(self):
		
		tok = self.current_tok

		if tok.type in (t_int, t_float):
			self.advance()
			return res.success(NumberNode(tok))

		return res.failure(error)

	def term(self):
		return self.bin_op(self.factor, (t_multplication, t_division))

	def expr(self):
		return self.bin_op(self.factor, (t_plus, t_minus))

	def bin_op(self, func, ops):
		res = ParseResult()
		left = res.register(func())
		if res.error: return res

		while self.current_tok.type in ops:
			op_tok = self.current_tok
			res.register(self.advance())
			right = res.register(func())
			if res.error: return res
			left = BinOpNode(left, op_tok, right)

		return res.success(left)

class RTResult:
	def __init__(self):
		self.value = None
		self.error = None

	def register(self, res):
		self.error = res.error
		return res.value

	def success(self, value):
		self.value = value
		return self

	def failure(self, error):
		self.error = error
		return self


class Number(Value):
	def __init__(self, value):
		super().__init__()
		self.value = value

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self

	def added_to(self, other):
		if isinstance(other, Number):
			return Number(self.value + other.value).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def subbed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value - other.value).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def multed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value * other.value).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def divided_by(self, other):
		if isinstance(other, Number):
			return Number(self.value / other.value).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)


class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos
		self.symbol_table = None

class Interpreter:
	def visit(self, node, context):
		method_name = f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)

	def no_visit_method(self, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined')
	
	def visit_NumberNode(self, node, context):
		return RTResult().success(
			Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
		)



	def visit_BinOpNode(self, node, context):

		left = res.register(self.visit(node.left_node, context))
		right = res.register(self.visit(node.right_node, context))

		if node.op_tok.type == t_plus:
			result, error = left.added_to(right)
		elif node.op_tok.type == t_minus:
			result, error = left.subbed_by(right)
		elif node.op_tok.type == t_multplication:
			result, error = left.multed_by(right)
		elif node.op_tok.type == t_division:
			result, error = left.dived_by(right)


		return result

	def visit_UnaryOpNode(self, node, context):
		number = self.visit(node.left_node, context)
		if res.error: return res


		if node.op_tok.type == t_minus:
			number, error = number.multed_by(Number(-1))
		return numbers




def run(fn, text):
	# Generate tokens
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_tokens()
	if error: return None, error
	
	# Generate AST
	parser = Parser(tokens)
	ast = parser.parse()
	



	return ast.node, ast.error