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
t_pow				= 'POW'
t_identifier		= 'IDENTIFIER'
t_keyword			= 'KEYWORD'
t_equals			= 'EE'
t_nEquals			= 'NN'
t_lessThan			= 'LT'
t_greatThan			= 'GT'
t_lessThanEq		= 'LTE'
t_greatThanEq		= 'GTE'



KEYWORDS = [
	'VAR',
	'AND',
	'OR',
]

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

	def notted(self):
		return Number(1 if self.value == 0 else 0).set_context(self.context), None

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
		self.fn = fn
		self.text = text
		self.pos = Position(-1, 0, -1, fn, text)
		self.current_pos = None
		self.advance()
	
	def advance(self):
		self.pos.advance(self.current_pos)
		self.current_pos = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

	def make_tokens(self):
		tokens = []

		while self.current_pos != None:
			if self.current_pos in ' \t':
				self.advance()
			elif self.current_pos == '+':
				tokens.append(Token(t_plus, pos_start=self.pos))
				self.advance()
			elif self.current_pos == '-':
				tokens.append(Token(t_minus, pos_start=self.pos))
				self.advance()
			elif self.current_pos == '*':
				tokens.append(Token(t_multplication, pos_start=self.pos))
				self.advance()
			elif self.current_pos == '/':
				tokens.append(Token(t_division, pos_start=self.pos))
				self.advance()
			elif self.current_pos == '^':
				tokens.append(Token(t_pow, pos_start=self.pos))
				self.advance()
			elif self.current_pos == '(':
				tokens.append(Token(t_lparen, pos_start=self.pos))
				self.advance()
			elif self.current_pos == ')':
				tokens.append(Token(t_rparen, pos_start=self.pos))
				self.advance()
			elif self.current_pos == '!':
				token, error = self.notEqual()
				if error: return [], error
				tokens.append(token)
			elif self.current_pos == '=':
				tokens.append(self.equal())
			elif self.current_pos == '<':
				tokens.append(self.lessThan())
			elif self.current_pos == '>':
				tokens.append(self.greaterThan())
			else:
				pos_start = self.pos.copy()
				char = self.current_pos
				self.advance()
				return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

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

	def chooseIdentifier(self):
		id_str = ''
		pos_start = self.pos.copy()

		while self.current_pos != None and self.current_pos in LETTERS_DIGITS + '_':
			id_str += self.current_pos
			self.advance()

		tok_type = t_keyword if id_str in KEYWORDS else t_identifier
		return Token(tok_type, id_str, pos_start, self.pos)

	def notEqual(self):
		pos_start = self.pos.copy()
		self.advance()

		if self.current_pos == '=':
			self.advance()
			return Token(t_nEquals, pos_start=pos_start, pos_end=self.pos), None

		self.advance()
		return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")
	
	def equal(self):
		tok_type = t_EQ
		pos_start = self.pos.copy()
		self.advance()

		if self.current_pos == '=':
			self.advance()
			tok_type = t_EE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

	def lessThan(self):
		tok_type = t_LT
		pos_start = self.pos.copy()
		self.advance()

		if self.current_pos == '=':
			self.advance()
			tok_type = t_LTE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

	def greaterThan(self):
		tok_type = t_GT
		pos_start = self.pos.copy()
		self.advance()

		if self.current_pos == '=':
			self.advance()
			tok_type = t_GTE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)


	
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
   
	def atom(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (t_int, t_float):
			res.register_advancement()
			self.advance()
			return res.success(NumberNode(tok))

		elif tok.type == t_lparen:
			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			if self.current_tok.type == t_rparen:
				res.register_advancement()
				self.advance()
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ')'"
				))

	def factor(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (t_plus, t_minus):
			res.register_advancement()
			self.advance()
			factor = res.register(self.factor())
			if res.error: return res
			return res.success(UnaryOpNode(tok, factor))

		return self.power()

	def power(self):
		return self.bin_op(self.call, (t_pow, ), self.factor)

	def term(self):
		return self.bin_op(self.factor, (t_multplication, t_division))

	def expr(self):
		res = ParseResult()

		if self.current_tok.matches(t_keyword, 'VAR'):
			res.register_advancement()
			self.advance()

			if self.current_tok.type != t_identifier:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected identifier"
				))

			var_name = self.current_tok
			res.register_advancement()
			self.advance()

			if self.current_tok.type != t_equals:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected '='"
				))

			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			return res.success(VarAssignNode(var_name, expr))

		node = res.register(self.bin_op(self.comp_expr, ((t_keyword, 'AND'), (t_keyword, 'OR'))))

		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected 'VAR', int, float, identifier, '+', '-', '('"
			))

		return res.success(node)

	def comp_expr(self):
		res = ParseResult()
		node = res.register(self.bin_op(self.arith_expr, (t_equals, t_nEquals, t_lessThan, t_greatThan, t_lessThanEq, t_greatThanEq)))
		
		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected int, float, identifier, '+', '-', '(''"
			))

		return res.success(node)

	def arith_expr(self):
		return self.bin_op(self.term, (t_plus, t_minus))

	def bin_op(self, func_a, ops, func_b=None):
		if func_b == None:
			func_b = func_a
		
		res = ParseResult()
		left = res.register(func_a())
		if res.error: return res

		while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()
			right = res.register(func_b())
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

	def multiply_by(self, other):
		if isinstance(other, Number):
			return Number(self.value * other.value).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def divided_by(self, other):
		if isinstance(other, Number):
			if other.value == 0:
				return None, RTError(
					other.pos_start, other.pos_end,
					'Can not divide anything by 0',
					self.context
				)

			return Number(self.value / other.value).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def powered_by(self, other):
		if isinstance(other, Number):
			return Number(self.value ** other.value).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def get_comparison_eq(self, other):
		if isinstance(other, Number):
			return Number(int(self.value == other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def get_comparison_ne(self, other):
		if isinstance(other, Number):
			return Number(int(self.value != other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def get_comparison_lt(self, other):
		if isinstance(other, Number):
			return Number(int(self.value < other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def get_comparison_gt(self, other):
		if isinstance(other, Number):
			return Number(int(self.value > other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def get_comparison_lte(self, other):
		if isinstance(other, Number):
			return Number(int(self.value <= other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def get_comparison_gte(self, other):
		if isinstance(other, Number):
			return Number(int(self.value >= other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def anded_by(self, other):
		if isinstance(other, Number):
			return Number(int(self.value and other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def ored_by(self, other):
		if isinstance(other, Number):
			return Number(int(self.value or other.value)).set_context(self.context), None
		else:
			return None, Value.illegal_operation(self, other)

	def copy(self):
		copy = Number(self.value)
		copy.set_pos(self.pos_start, self.pos_end)
		copy.set_context(self.context)
		return copy


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

	def visit_VarAccessNode(self, node, context):
		res = RTResult()
		var_name = node.var_name_tok.value
		value = context.symbol_table.get(var_name)

		if not value:
			return res.failure(RTError(
				node.pos_start, node.pos_end,
				f"'{var_name}' is not defined",
				context
			))

		value = value.copy().set_pos(node.pos_start, node.pos_end)
		return res.success(value)

	def visit_VarAssignNode(self, node, context):
		res = RTResult()
		var_name = node.var_name_tok.value
		value = res.register(self.visit(node.value_node, context))
		if res.error: return res

		context.symbol_table.set(var_name, value)
		return res.success(value)


	def visit_BinOpNode(self, node, context):
		res = RTResult()
		left = res.register(self.visit(node.left_node, context))
		if res.error: return res
		right = res.register(self.visit(node.right_node, context))
		if res.error: return res

		if node.op_tok.type == t_plus:
			result, error = left.added_to(right)
		elif node.op_tok.type == t_minus:
			result, error = left.subbed_by(right)
		elif node.op_tok.type == t_multplication:
			result, error = left.multiply_by(right)
		elif node.op_tok.type == t_division:
			result, error = left.divided_by(right)
		elif node.op_tok.type == t_pow:
			result, error = left.powered_by(right)
		elif node.op_tok.type == t_equals:
			result, error = left.get_comparison_eq(right)
		elif node.op_tok.type == t_nEquals:
			result, error = left.get_comparison_ne(right)
		elif node.op_tok.type == t_lessThan:
			result, error = left.get_comparison_lt(right)
		elif node.op_tok.type == t_greatThan:
			result, error = left.get_comparison_gt(right)
		elif node.op_tok.type == t_lessThanEq:
			result, error = left.get_comparison_lte(right)
		elif node.op_tok.type == t_greatThanEq:
			result, error = left.get_comparison_gte(right)
		elif node.op_tok.matches(t_keyword, 'AND'):
			result, error = left.anded_by(right)
		elif node.op_tok.matches(t_keyword, 'OR'):
			result, error = left.ored_by(right)


		
		if error:
			return res.failure(error)
		else:
			return res.success(result.set_pos(node.pos_start, node.pos_end))

	def visit_UnaryOpNode(self, node, context):
		res = RTResult()
		number = res.register(self.visit(node.node, context))
		if res.error: return res

		error = None

		if node.op_tok.type == t_minus:
			number, error = number.multed_by(Number(-1))
		elif node.op_tok.matches(t_keyword, 'NOT'):
			number, error = number.notted()

		if error:
			return res.failure(error)
		else:
			return res.success(number.set_pos(node.pos_start, node.pos_end))




def run(fn, text):
	# Generate tokens
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_tokens()
	if error: return None, error
	
	# Generate AST
	parser = Parser(tokens)
	ast = parser.parse()
	



	return ast.node, ast.error