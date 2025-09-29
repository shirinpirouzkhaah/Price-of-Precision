import subprocess

path_idioms_file = './utils/idioms.csv'
class Abstracter:
    """Class to perform abstraction on code files"""

    def __init__(self, path_idioms_file):
        if path_idioms_file is None:
            # no idioms
            self.path_idioms_file = [path_idioms_file]
        else:
            self.path_idioms_file = path_idioms_file

    def abstract(self, path_file1, path_file2, path_abs_file1, path_abs_file2):
        Abstracter.write_abstraction_file(path_file1, path_file2, path_abs_file1, path_abs_file2, self.path_idioms_file)
        subprocess.run('./utils/abstract.sh')
        # check parsing error during abstraction
        if not (Abstracter.check_abstraction(path_abs_file1) or not Abstracter.check_abstraction(path_abs_file2)):
            return False
        Abstracter.fix_abstraction(path_abs_file1, path_abs_file2)
        return True

    def abstract_single(self, path_file1, path_file2, path_abs_file1, path_abs_file2):
        Abstracter.write_abstraction_file_single(path_file1, path_abs_file1, self.path_idioms_file)
        subprocess.run('./utils/abstract.sh')

        Abstracter.write_abstraction_file_single(path_file2, path_abs_file2, self.path_idioms_file)
        subprocess.run('./utils/abstract.sh')
        # check parsing error during abstraction
        if not (Abstracter.check_abstraction(path_abs_file1) or not Abstracter.check_abstraction(path_abs_file2)):
            return False
        # Abstracter.fix_abstraction(path_abs_file1, path_abs_file2)
        return True
    
    
    def abstract_single_diff(self, path_file1, path_abs_file1):
        Abstracter.write_abstraction_file_single(path_file1, path_abs_file1, self.path_idioms_file)
        subprocess.run('./utils/abstract.sh')

        
        # check parsing error during abstraction
        if not Abstracter.check_abstraction(path_abs_file1):
            return False
        # Abstracter.fix_abstraction(path_abs_file1, path_abs_file2)
        return True
    
    @staticmethod
    def reconstruct_path(path, abs_map_folder=0):
        path_elements = path.split('/')
        if '.' in path_elements[-1]:
            filename = path_elements[-1].split('.')[0]
        else:
            filename = path_elements[-1]
        new_path = '/'
        for elem in path_elements[1:-1]:
            new_path += elem + '/'
        if abs_map_folder == 0:
            new_path += filename + '_abs'
        elif abs_map_folder == 1:
            new_path += filename + '.map'
        return new_path

    @staticmethod
    def write_abstraction_file(path_file1, path_file2, path_abs_file1, path_abs_file2, path_idioms):
        f = open('./utils/abstract.sh', 'w')
        f.close()
        subprocess.run('chmod a+x ./utils/abstract.sh', shell=True)
        f = open('./utils/abstract.sh', 'w')
        f.write('#!/usr/bin/env bash\n')
        f.write(
            'java -jar ./utils/src2abs-0.1-jar-with-dependencies.jar '
            'pair method ')
        f.write(path_file1 + ' ')
        f.write(path_file2 + ' ')
        f.write(path_abs_file1 + ' ')
        f.write(path_abs_file2 + ' ')
        f.write(path_idioms)
        f.close()

    @staticmethod
    def write_abstraction_file_single(path_file1, path_abs_file1, path_idioms):
        f = open('./utils/abstract.sh', 'w')
        f.close()
        subprocess.run('chmod a+x ./utils/abstract.sh', shell=True)
        f = open('./utils/abstract.sh', 'w')
        f.write('#!/usr/bin/env bash\n')
        f.write(
            'java -jar ./utils/src2abs-0.1-jar-with-dependencies.jar '
            'single method ')
        f.write(path_file1 + ' ')
        f.write(path_abs_file1 + ' ')
        f.write(path_idioms)
        f.close()
    

    @staticmethod
    def check_abstraction(path_file):
        f = open(path_file, 'r')
        abs_file = f.read()
        f.close()
        if abs_file == '<ERROR>':
            return False
        return True

    @staticmethod
    def fix_abstraction(path_abs_file1, path_abs_file2):
        # path_map = Abstracter.reconstruct_path(path_abs_file1, 1)
        values, keys = Abstracter.load_map('./utils/before_abs.map')
        variables, types, methods, characters, strings, annotations, floats, integers = \
            Abstracter.separate_different_token(values, keys)
        strings = Abstracter.fix_strings(strings)
        old_tokens = []
        new_tokens = []
        new_types, types_to_modify = Abstracter.create_new_types(types)
        count_types = len(types)
        if len(types_to_modify) != 0:
            count_types, new_types, old_tokens, new_tokens = Abstracter.modify_types(
                count_types, types_to_modify, new_types, old_tokens, new_tokens)
        count_types, new_types, new_variables, old_tokens, new_tokens = Abstracter.create_new_variables(
            variables, count_types, new_types, old_tokens, new_tokens)
        # path_data = Abstracter.reconstruct_path(path_abs_file1, 2)
        Abstracter.save_dictionary(new_variables, new_types, methods, strings, characters,
                                   annotations, floats, integers)
        if len(old_tokens) != 0:
            Abstracter.modify_abstraction(path_abs_file1, old_tokens, new_tokens)
            Abstracter.modify_abstraction(path_abs_file2, old_tokens, new_tokens)
            
            
    @staticmethod
    def fix_abstraction_diff(path_abs_file1):
        # path_map = Abstracter.reconstruct_path(path_abs_file1, 1)
        values, keys = Abstracter.load_map('./utils/before_abs.map')
        variables, types, methods, characters, strings, annotations, floats, integers = \
            Abstracter.separate_different_token(values, keys)
        strings = Abstracter.fix_strings(strings)
        old_tokens = []
        new_tokens = []
        new_types, types_to_modify = Abstracter.create_new_types(types)
        count_types = len(types)
        if len(types_to_modify) != 0:
            count_types, new_types, old_tokens, new_tokens = Abstracter.modify_types(
                count_types, types_to_modify, new_types, old_tokens, new_tokens)
        count_types, new_types, new_variables, old_tokens, new_tokens = Abstracter.create_new_variables(
            variables, count_types, new_types, old_tokens, new_tokens)
        # path_data = Abstracter.reconstruct_path(path_abs_file1, 2)
        Abstracter.save_dictionary(new_variables, new_types, methods, strings, characters,
                                   annotations, floats, integers)
        if len(old_tokens) != 0:
            Abstracter.modify_abstraction(path_abs_file1, old_tokens, new_tokens)
            
    @staticmethod
    def load_map(path):
        f = open(path)
        rows = [line[:-1] for line in f]
        f.close()
        keys = []
        values = []
        for row in rows:
            row_elements = row.split(' : ')
            values.append(row_elements[0])
            if len(row_elements) == 2:
                keys.append(row_elements[1])
            else:
                key = ''
                for elem in row_elements[1:-1]:
                    key += elem + ' : '
                key += row_elements[-1]
                keys.append(key)
        return values, keys

    @staticmethod
    def separate_different_token(values, keys):
        variables = []
        types = []
        methods = []
        characters = []
        strings = []
        annotations = []
        floats = []
        integers = []

        for i in range(len(values)):
            if values[i].startswith('VAR'):
                variables.append([values[i], keys[i]])
            elif values[i].startswith('TYPE'):
                types.append([values[i], keys[i]])
            elif values[i].startswith('METHOD'):
                methods.append([values[i], keys[i]])
            elif values[i].startswith('CHAR'):
                characters.append([values[i], keys[i]])
            elif values[i].startswith('STRING'):
                strings.append([values[i], keys[i]])
            elif values[i].startswith('ANNOTATION'):
                annotations.append([values[i], '@' + keys[i]])
            elif values[i].startswith('FLOAT'):
                floats.append([values[i], keys[i]])
            elif values[i].startswith('INT'):
                integers.append([values[i], keys[i]])
        return variables, types, methods, characters, strings, annotations, floats, integers

    @staticmethod
    def fix_strings(strings):
        for elem in strings:
            elem[1] = elem[1].replace('<DOUBLE_SLASH>', '//')
            elem[1] = elem[1].replace('<AT>', '@')
        return strings

    @staticmethod
    def create_new_types(types):
        new_types = []
        types_to_modify = []
        for i in range(len(types)):
            if '.' in types[i][1]:
                types_to_modify.append(types[i])
            else:
                new_types.append(types[i])
        return new_types, types_to_modify

    @staticmethod
    def modify_types(count_types, types_to_modify, new_types, old_tokens, new_tokens):
        for i in range(len(types_to_modify)):
            old_tokens.append(types_to_modify[i][0])
            new_token = ''
            splitted_types = types_to_modify[i][1].split('.')
            for k in range(len(splitted_types)):
                current_type = splitted_types[k]
                flag = False
                j = 0
                while j < len(new_types) and not flag:
                    if new_types[j][1] == current_type:
                        flag = True
                        new_token += new_types[j][0]
                    else:
                        j += 1
                if not flag:
                    count_types += 1
                    new_type_token = 'TYPE_' + str(count_types)
                    new_types.append([new_type_token, current_type])
                    new_token += new_type_token
                if k != len(splitted_types) - 1:
                    new_token += ' . '
            new_tokens.append(new_token)
        return count_types, new_types, old_tokens, new_tokens

    @staticmethod
    def create_new_variables(variables, count_types, new_types, old_tokens, new_tokens):
        new_variables = []
        for i in range(len(variables)):
            var = variables[i][1]
            if len(var) == 1:
                new_variables.append(variables[i])
                continue
            if var[0] != var[0].lower() and var != var.upper():
                old_token = variables[i][0]
                old_tokens.append(old_token)
                flag = True
                if len(new_types) != 0:
                    for j in range(len(new_types)):
                        if new_types[j][1] == var:
                            new_token = new_types[j][0]
                            new_tokens.append(new_token)
                            flag = False
                if flag:
                    count_types += 1
                    new_token = 'TYPE_' + str(count_types)
                    new_types.append([new_token, var])
                    new_tokens.append(new_token)
            else:
                new_variables.append(variables[i])
        return count_types, new_types, new_variables, old_tokens, new_tokens


    @staticmethod
    def save_dictionary(new_variables, new_types, methods, strings, characters,
                        annotations, floats, integers):
        # SAVE DICTIONARY (in a more readable form)
        f = open('./utils/dictionary.txt', 'w')
        # add strings
        for i in range(len(strings)):
            line = strings[i][0] + " : " + strings[i][1]
            f.write(line)
            f.write('\n')
        # add variables
        for i in range(len(new_variables)):
            line = new_variables[i][0] + " : " + new_variables[i][1]
            f.write(line)
            f.write('\n')
        # add types
        for i in range(len(new_types)):
            line = new_types[i][0] + " : " + new_types[i][1]
            f.write(line)
            f.write('\n')
        # add methods
        for i in range(len(methods)):
            line = methods[i][0] + " : " + methods[i][1]
            f.write(line)
            f.write('\n')
        # add char
        for i in range(len(characters)):
            line = characters[i][0] + " : " + characters[i][1]
            f.write(line)
            f.write('\n')
        # add annotations
        for i in range(len(annotations)):
            line = annotations[i][0] + " : " + annotations[i][1]
            f.write(line)
            f.write('\n')
        # add floats
        for i in range(len(floats)):
            line = floats[i][0] + " : " + floats[i][1]
            f.write(line)
            f.write('\n')
        # add integers
        for i in range(len(integers)):
            line = integers[i][0] + " : " + integers[i][1]
            f.write(line)
            f.write('\n')
        f.close()

    @staticmethod
    def modify_abstraction(path, old_tokens, new_tokens):
        f = open(path, 'r')
        m = f.read()
        f.close()
        m_tokens = m.split()
        new_m = ''
        for token in m_tokens:
            if token in old_tokens:
                index = old_tokens.index(token)
                new_m += new_tokens[index] + ' '
            else:
                new_m += token + ' '
        f = open(path, 'w')
        f.write(new_m)
        f.close()

