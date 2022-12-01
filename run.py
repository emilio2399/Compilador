import calc

while True:
	text = input('calculadora > ')
	result, error = calc.run('<stdin>', text)

	if error: print(error.as_string())
	elif result: print(result)