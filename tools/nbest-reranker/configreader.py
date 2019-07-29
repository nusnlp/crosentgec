''' File to parse config files
'''

def parse_ini(ini_path):
	out = []
	with open(ini_path, 'r') as ini_file:
		section = '[nil]'
		for line in ini_file:
			line = line.strip()
			if line.startswith('['):
				section = line
			elif section == '[weight]' and line != '':
				if line.startswith('UnknownWordPenalty0= '):
					out.append('UnknownWordPenalty0 UNTUNEABLE')
				else:
					out.append(line)
	return out
