import lizard
import pandas as pd
import difflib
import numpy as np
import re
import traceback 




class Analyzer1:
    """Class to analyze data from GitHub
       The data must be organized in a pandas DataFrame"""

    def __init__(self, data, data_type):
        self.data = data
        self.data_type = data_type
        # data info
        self.before_equal_after = 0
        self.comm_to_comm = 0
        self.duplicates = 0
        self.left_side_cases = 0
        self.nan_data = 0
        self.no_comment = 0
        self.no_marked = 0
        self.no_method_after = 0
        self.no_method_before = 0
        self.no_valid_ref = 0
        self.owner_comments = 0



    @staticmethod
    def diff_extraction(file1, file2):
        
        with open(file1, 'r') as file:
            before = file.readlines()
        
        with open(file2, 'r') as file:
            after = file.readlines()
            
        diff = list(difflib.unified_diff(before, after, lineterm=""))
        diff_hunk = ''.join(diff)
                
        return diff_hunk


    @staticmethod
    def split_diff_hunk(diff_hunk):
        """
        Splits a unified diff string into a dictionary keyed by hunk headers.
        Each value is the body (code lines) under that header, including the header line itself.
        If a header has no body, it maps to just the header line.
    
        Returns:
            Dict[str, str]: {header: hunk_with_header}
        """
        pattern = re.compile(r"(@@ -\d+,\d+ \+\d+,\d+ @@)")
        pieces = pattern.split(diff_hunk)[1:]
    
        hunk_dict = {}
        for i in range(0, len(pieces), 2):
            header = pieces[i]
            body = pieces[i + 1].lstrip("\n") if i + 1 < len(pieces) else ""
            full_hunk = header + "\n" + body if body else header
            hunk_dict[header] = full_hunk
    
        return hunk_dict


    
    
    @staticmethod
    def extract_old_file_and_target_from_diff_hunk(diff_hunk_piece):
        """
        Extracts old and new versions from a unified diff hunk,
        preserving '+' and '-' for added/removed lines,
        and ' ' for context lines.
    
        Args:
            diff_hunk_piece (str): A single diff hunk including the header line.
    
        Returns:
            Tuple[str, str]: (old version with '-', new version with '+')
        """
        old = []
        new = []
    
        lines = diff_hunk_piece.strip().split('\n')
    
        if not lines or not lines[0].startswith("@@"):
            raise ValueError("No hunk header found in the diff piece.")
    
        for line in lines[1:]:
            if line.startswith(' '):
                old.append(line)
                new.append(line)
            elif line.startswith('-'):
                old.append(line[1:])
                new.append(line)
            elif line.startswith('+'):
                new.append(line)
            else:
                # Optionally log malformed lines
                continue
    
        return "\n".join(old), "\n".join(new)



    def remove_duplicates(self):
        start_len = len(self.data)
    
        # Identify columns with unhashable types (e.g., list)
        hashable_cols = []
        for col in self.data.columns:
            sample = self.data[col].dropna().iloc[0] if not self.data[col].dropna().empty else None
            if not isinstance(sample, list):  # extendable if you find other unhashable types
                hashable_cols.append(col)
            else:
                print(f"⚠️ Skipping column '{col}' in duplicate detection due to unhashable type: list")
    
        # Drop duplicates using only hashable columns
        no_duplicates = self.data.drop_duplicates(subset=hashable_cols)
    
        self.data = no_duplicates
        self.duplicates = start_len - len(self.data)


    # def remove_duplicates(self):
    #     start_len = len(self.data)
    #     no_duplicates = self.data.drop_duplicates()
    #     self.data = no_duplicates
    #     self.duplicates = start_len - len(self.data)

    def remove_owner_comments(self):
        if self.data_type == 'Gerrit':
            new_data = Analyzer1.owner_gerrit(self.data)
        else:  # GitHub
            new_data = Analyzer1.owner_github(self.data)
        self.owner_comments = len(self.data) - len(new_data)
        self.data = new_data 

    @staticmethod
    def owner_gerrit(df):
        return df[df['change_owner'] != 'owner']

    @staticmethod
    def owner_github(df):
        return df[df['user_id'] != df['owner_id']]

    def remove_left_side(self):
        if self.data_type == 'GitHub':
            start_len = len(self.data)
            self.data = self.data[self.data['side'] != 'LEFT']
            self.left_side_cases = start_len - len(self.data)


    def remove_nan_data(self):
        start_len = len(self.data)
        if self.data_type == 'Gerrit':
            new_df = self.data.dropna(subset=['message', 'file_content_before', 'file_content_after'])
        else:  # GitHub
            new_df = self.data.dropna(subset=['message', 'file_content_while', 'file_content_after'])
        self.data = new_df.reset_index(drop=True)
        self.nan_data = start_len - len(self.data)
    
    
    def remove_nan_data_from_method_dfr(self):
        start_len = len(self.method_dfr)
    
        # Drop rows with NaNs first
        cleaned_df = self.method_dfr.dropna(subset=['before_marked', 'after', 'comment']).copy()
    
        # Use .loc to avoid SettingWithCopyWarning
        for col in ['before_marked', 'after', 'comment']:
            cleaned_df.loc[:, col] = cleaned_df[col].astype(str)
    
        # Drop rows that are empty or whitespace only
        cleaned_df = cleaned_df[
            (cleaned_df['before_marked'].str.strip() != '') &
            (cleaned_df['after'].str.strip() != '') &
            (cleaned_df['comment'].str.strip() != '')
        ]
    
        self.method_dfr = cleaned_df.reset_index(drop=True)
        self.nan_data = start_len - len(self.method_dfr)


    
    
    def remove_nan_data_from_diff_dfr(self):
        start_len = len(self.diff_dfr)
    
        # Drop rows with NaNs first
        cleaned_df = self.diff_dfr.dropna(subset=['comment', 'old', 'new']).copy()
    
        # Use .loc to avoid SettingWithCopyWarning
        for col in ['comment', 'old', 'new']:
            cleaned_df.loc[:, col] = cleaned_df[col].astype(str)
    
        # Drop rows where any field is empty or just whitespace
        cleaned_df = cleaned_df[
            (cleaned_df['comment'].str.strip() != '') &
            (cleaned_df['old'].str.strip() != '') &
            (cleaned_df['new'].str.strip() != '')
        ]
    
        self.diff_dfr = cleaned_df.reset_index(drop=True)
        self.nan_data = start_len - len(self.diff_dfr)





    #more than one comment for same before+after, only keep earliest comment     
    """
    method A + C1 -->A' keep
    method A + C2 -->A' remove
    method A + C3 -->A' remove
    method A + C4 -->A' remove
    """
    def keep_earliest_comment_methodLevel(self, method_dfr):
        # Use unified timestamp field
        method_dfr['comment_time'] = pd.to_datetime(method_dfr['creation_time'])
    
        # Group by transformation of before+after
        method_dfr['combined_hash'] = (method_dfr['before'] + method_dfr['after']).apply(hash)
    
        idx_to_keep = method_dfr.groupby('combined_hash')['comment_time'].idxmin().values
        result_df = method_dfr.loc[idx_to_keep].copy()
    
        return result_df.drop(columns=['combined_hash', 'comment_time']).reset_index(drop=True)


     


    #more than one comment for same before+after, instead of keeping the earliest, remove all the records
    """
    method A + C1 -->A' remove
    method A + C2 -->A' remove
    method A + C3 -->A' remove
    method A + C4 -->A' remove
    """
    def remove_multiple_comments_methodLevel(self, method_dfr):
        method_dfr['combined_hash'] = (method_dfr['before'] + method_dfr['after']).apply(hash)
        unique_hashes = method_dfr['combined_hash'].value_counts() == 1
        filtered_df = method_dfr[method_dfr['combined_hash'].isin(unique_hashes[unique_hashes].index)].copy()
    
        return filtered_df.drop(columns=['combined_hash']).reset_index(drop=True)



     
    # one same comment to several different records of befor+after (EMSE MAJOR REVISION ASKED FOR THIS), keep only the row with the earliest timestamp
    """
    method A + C1 -->A' keep
    method B + C1 -->B' remove
    method C + C1 -->C' remove
    method D + C1 -->D' remove
    """
    def remove_shared_comments_methodLevel(self, method_dfr):
        method_dfr['comment_time'] = pd.to_datetime(method_dfr['creation_time'])
    
        method_dfr['comment_hash'] = method_dfr['comment'].apply(hash)
        method_dfr['methods_hash'] = (method_dfr['before'] + method_dfr['after']).apply(hash)
    
        grouped = method_dfr.groupby('comment_hash')['methods_hash'].nunique().reset_index(name='unique_method_count')
        shared_comment_hashes = grouped[grouped['unique_method_count'] > 1]['comment_hash']
    
        shared_df = method_dfr[method_dfr['comment_hash'].isin(shared_comment_hashes)].copy()
        non_shared_df = method_dfr[~method_dfr['comment_hash'].isin(shared_comment_hashes)].copy()
    
        idx_to_keep = shared_df.groupby('comment_hash')['comment_time'].idxmin().values
        earliest_shared_df = shared_df.loc[idx_to_keep]
    
        final_df = pd.concat([non_shared_df, earliest_shared_df], ignore_index=True)
    
        return final_df.drop(columns=['comment_hash', 'methods_hash', 'comment_time']).reset_index(drop=True)




    #more than one comment for same diff hunk, only keep earliest comment   
    """
    diff A + C1 -->A' keep
    diff A + C2 -->A' remove
    diff A + C3 -->A' remove
    diff A + C4 -->A' remove
    """
    def keep_earliest_comment_diffLevel(self, diff_dfr):
        diff_dfr['comment_time'] = pd.to_datetime(diff_dfr['creation_time'])
        diff_dfr['diff_hunk_hash'] = diff_dfr['old'].astype(str).apply(hash)
    
        idx_to_keep = diff_dfr.groupby('diff_hunk_hash')['comment_time'].idxmin().values
        result_df = diff_dfr.loc[idx_to_keep].copy()
    
        return result_df.drop(columns=['diff_hunk_hash', 'comment_time']).reset_index(drop=True)

    
    
    
    #more than one comment for same diff hunk, instead of keeping the earliest, remove all the records
    """
    diff A + C1 -->A' remove
    diff A + C2 -->A' remove
    diff A + C3 -->A' remove
    diff A + C4 -->A' remove
    """
    def remove_multiple_comments_diffLevel(self, diff_dfr):
        diff_dfr['diff_hunk_hash'] = diff_dfr['old'].astype(str).apply(hash)
        unique_hashes = diff_dfr['diff_hunk_hash'].value_counts() == 1
        filtered_df = diff_dfr[diff_dfr['diff_hunk_hash'].isin(unique_hashes[unique_hashes].index)].copy()
    
        return filtered_df.drop(columns=['diff_hunk_hash']).reset_index(drop=True)



    
    # one same comment to several different diff hunks
    """
    diff A + C1 -->A' keep
    diff B + C1 -->B' remove
    diff C + C1 -->C' remove
    diff D + C1 -->D' remove
    """
    
    def remove_shared_comments_diffLevel(self, diff_dfr):
        diff_dfr['comment_time'] = pd.to_datetime(diff_dfr['creation_time'])
    
        diff_dfr['comment_hash'] = diff_dfr['comment'].apply(hash)
        diff_dfr['old_str'] = diff_dfr['old'].astype(str)
    
        grouped = diff_dfr.groupby('comment_hash')['old_str'].nunique().reset_index(name='unique_diff_count')
        shared_comment_hashes = grouped[grouped['unique_diff_count'] > 1]['comment_hash']
    
        shared_df = diff_dfr[diff_dfr['comment_hash'].isin(shared_comment_hashes)].copy()
        non_shared_df = diff_dfr[~diff_dfr['comment_hash'].isin(shared_comment_hashes)].copy()
    
        idx_to_keep = shared_df.groupby('comment_hash')['comment_time'].idxmin().values
        earliest_shared_df = shared_df.loc[idx_to_keep]
    
        final_df = pd.concat([non_shared_df, earliest_shared_df], ignore_index=True)
    
        return final_df.drop(columns=['comment_hash', 'old_str', 'comment_time']).reset_index(drop=True)



        

    def analyze_method_comments_Data(self):
        
        df = self.data
        drop_indices = []
        
        for i in range(len(df)):
            
            if df.iloc[i]['owner_comment'] == True:
                drop_indices.append(i)
                
            if df.iloc[i]['comment_to_comment'] == True:
                drop_indices.append(i)
            # check comment
            
            if df.iloc[i]['method_flag_linked'] == False:
                drop_indices.append(i)
            
            comm = str(df.iloc[i]['comment']).strip()
            if len(comm) == 0:
                drop_indices.append(i)
           
        # Drop the rows with indices in drop_indices list
        dfr = df.drop(drop_indices)
        dfr = dfr.reset_index(drop=True)
        #shared_method_removed = Analyzer1.remove_multiple_comments_methodLevel(dfr)
        
        return dfr
        

    
    def analyze_diff_comments_Data(self):
        
        df = self.data
        drop_indices = []
        
        for i in range(len(df)):
            
            if df.iloc[i]['owner_comment'] == True:
                drop_indices.append(i)
                
            if df.iloc[i]['comment_to_comment'] == True:
                drop_indices.append(i)
            # check comment
            
            
            comm = str(df.iloc[i]['comment']).strip()
            if len(comm) == 0:
                drop_indices.append(i)
                
           
        # Drop the rows with indices in drop_indices list
        dfr = df.drop(drop_indices)
        dfr = dfr.reset_index(drop=True)
        
        #shared_diff_keep_earliest_comment_dfr = Analyzer1.keep_earliest_comment_diffLevel(dfr)
        #shared_comments_removed_dfr = Analyzer1.remove_shared_comments_diffLevel(shared_diff_keep_earliest_comment_dfr)
        

        return dfr
    
    
    def analyze_equal_methods(self):
        df = self.data
        drop_indices = []
        
        for i in range(len(df)):
            before = df.iloc[i]['before']
            after = df.iloc[i]['after']
            
            # Check if 'before' or 'after' is NaN or an empty string
            if pd.isna(before) or pd.isna(after) or before == "" or after == "":
                drop_indices.append(i)
                continue 
                
            # Check if 'before' is equal to 'after'
            if before == after:
                drop_indices.append(i)
            
        # Drop the rows with indices in drop_indices list
        dfr = df.drop(drop_indices)
        
        # Reset the index
        dfr = dfr.reset_index(drop=True)
    
        return dfr

    
    def analyze_empty_diff(self):
        df = self.data
        drop_indices = []
    
        for i in range(len(df)):
            old = df.iloc[i]['old']
            new = df.iloc[i]['new']
    
            # Drop if either old or new is NaN
            if pd.isna(old) or pd.isna(new):
                drop_indices.append(i)
            # Drop if either old or new is empty or whitespace-only
            elif not str(old).strip() or not str(new).strip():
                drop_indices.append(i)
    
        # Drop the rows with indices in drop_indices list
        dfr = df.drop(drop_indices).reset_index(drop=True)
    
        return dfr


    @staticmethod
    def get_info(df, index, data_type):
        if data_type == 'Gerrit':
            id_ref, num_ref, ref = Analyzer1.get_info_gerrit(df, index)
        else:
            id_ref, num_ref, ref = Analyzer1.get_info_github(df, index)
        filename = df.iloc[index]['filename']
        return id_ref, num_ref, ref, filename

    @staticmethod
    def get_info_github(df, index):
        pull_id = df.iloc[index]['pull_id']
        pull_num = df.iloc[index]['pull_number']
        if str(df.iloc[index]['original_start_line']) != 'nan':
            ref = [int(df.iloc[index]['original_start_line']),
                   df.iloc[index]['original_line']]
        else:
            ref = [df.iloc[index]['original_line'], df.iloc[index]['original_line']] 
        return pull_id, pull_num, ref

    @staticmethod
    def get_info_gerrit(df, index):
        change_id = df.iloc[index]['change_id']
        rev_num = df.iloc[index]['revision_number']
        ref = []
        start_line = df.iloc[index]['comment_start_line']
        line = df.iloc[index]['line']
        end_line = df.iloc[index]['comment_end_line']
        start_char = df.iloc[index]['start_character']
        end_char = df.iloc[index]['end_character']
        if start_line != 0:
            ref.append(start_line)
        else:
            ref.append(line)
        if end_line != 0:
            ref.append(end_line)
        else:
            ref.append(ref[0])
        if start_char != 0 and end_char != 0:
            ref.append(start_char)
            ref.append(end_char)
        return change_id, rev_num, ref



    @staticmethod
    def save_temp_code(df, index, data_type):
        before_path = 'before.java'
        after_path = 'after.java'
        beforePR_path = 'beforePR.java'
        
        if data_type == 'GitHub':
            with open(beforePR_path, 'w') as f:
                f.write(df.iloc[index].get('file_content_before', ' '))
        
        with open(before_path, 'w') as f:
            f.write(df.iloc[index].get('file_content_while' if data_type == 'GitHub' else 'file_content_before', ' '))
        
        with open(after_path, 'w') as f:
            f.write(df.iloc[index].get('file_content_after', ' '))
           
            

    @staticmethod
    def save_marked_diff(ref): 
        flag_marked = Analyzer1.mark_before_file_with_ref('before.java','marked_before.java', ref)
        
    
        org_hunk = Analyzer1.diff_extraction('before.java', 'after.java')
        if org_hunk:
            full_diff_path = "full_diff_file.diff"
            with open(full_diff_path, 'w') as diff_file:
                diff_file.write(org_hunk)
                
        else:
            full_diff_path = "full_diff_file.diff"
            with open(full_diff_path, 'w') as diff_file:
                diff_file.write("")  # Write an empty string explicitly
            
            
            
        marked_hunk = Analyzer1.diff_extraction('marked_before.java', 'after.java')
        
        
        
        hunk = Analyzer1.build_marked_hunk(org_hunk, marked_hunk)
        if hunk:
            full_diff_path = "full_marked_diff_file.diff"
            with open(full_diff_path, 'w') as diff_file:
                diff_file.write(hunk)
                
        else:
            full_diff_path = "full_marked_diff_file.diff"
            with open(full_diff_path, 'w') as diff_file:
                diff_file.write("")  # Write an empty string explicitly
            
        return flag_marked



    @staticmethod
    def mark_before_file_with_ref(before_path, marked_path, ref):
        """
        Annotate before.java line-level with START and END based on ref line range.
    
        Args:
            before_path (str): Path to the original before file.
            marked_path (str): Path to save the marked version.
            ref (list or tuple): A list of two integers [start_line, end_line] (1-based inclusive).
    
        Raises:
            ValueError: If ref is invalid or out of bounds.
        """
    
        ref_start, ref_end = sorted(ref[:2])
    
        with open(before_path, 'r') as f:
            code = [line.rstrip('\n') for line in f]
    
        total_lines = len(code)
        if not (1 <= ref_start <= total_lines) or not (1 <= ref_end <= total_lines):
            with open(marked_path, 'w') as f_out:
                f_out.write('\n'.join(code) + '\n')
            return False

        marked_lines = []
        flag_marked = False
    
        for idx, line in enumerate(code):
            line_num = idx + 1
    
            if line_num == ref_start == ref_end:
                marked_line = ' START ' + line + ' END '
                flag_marked = True
            elif line_num == ref_start:
                marked_line = ' START ' + line
            elif line_num == ref_end and not flag_marked:
                marked_line = line + ' END '
                flag_marked = True
            else:
                marked_line = line
    
            marked_lines.append(marked_line + '\n')
    
        with open(marked_path, 'w') as f:
            f.writelines(marked_lines)
    
        return flag_marked
    
    
    
    @staticmethod
    def build_marked_hunk(org_hunk, marked_hunk):
        """
        Cleans the marked_hunk by removing '+' or '-' prefixes from lines
        that only differ from org_hunk by START or END tags.
        
        Args:
            org_hunk (str): The original unified diff without START/END annotations.
            marked_hunk (str): The marked unified diff containing START/END annotations.
    
        Returns:
            str: A cleaned diff with only true differences kept marked.
        """
        if not marked_hunk.strip():
            return ''
    
        org_lines = org_hunk.splitlines()
        marked_lines = marked_hunk.splitlines()
    
        # Build a set of "context" lines (no prefix) from org_hunk
        context_lines = set()
        for line in org_lines:
            if line.startswith(' ') or (not line.startswith(('+', '-')) and not line.startswith('@@')):
                context_lines.add(line.lstrip(' '))
    
        cleaned_lines = []
    
        for line in marked_lines:
            if line.startswith('@@'):
                cleaned_lines.append(line)
                continue
    
            line_content = line[1:].strip() if line.startswith(('+', '-')) else line.strip()
    
            # Check if this is a line with <START> or <END> tag
            if ' START ' in line_content or ' END ' in line_content:
                # Remove the tags for matching
                tagless = line_content.replace(' START ', '').replace(' END ', '').strip()
    
                if tagless in context_lines:
                    # If it's just annotation, make it a context line
                    cleaned_lines.append(' ' + line_content)
                else:
                    # Real difference with annotation
                    cleaned_lines.append(line)
            else:
                # Not a tagged line, keep as is
                cleaned_lines.append(line)
    
        return '\n'.join(cleaned_lines) + '\n'

    @staticmethod
    def check_len_code(start_line, end_line):
        before_path = 'before.java'
        code_lines = [line for line in open(before_path)]
        if start_line > len(code_lines) or end_line > len(code_lines):
            return False
        return True


    def check_comment_to_comment(self, start_line, end_line):
        before_path = 'before.java'
        code_lines = [line.strip() for line in open(before_path)]
        k = start_line
        while k <= end_line:
            current_line = code_lines[k - 1]
            if len(current_line) == 0:
                k += 1
                continue
            if not current_line.startswith('/') and not current_line.startswith('*'):
                return False
            k += 1
            self.comm_to_comm += 1
            return True
        return False

    @staticmethod
    def search_before_method(ref):
        before_path = 'before.java'
        liz = lizard.analyze_file(before_path)
        method_found = []
        for liz_elem in liz.function_list:
            if (liz_elem.start_line <= ref[0]) and \
                    (liz_elem.end_line >= ref[1]):
                method_found.append(liz_elem)
        return method_found

    @staticmethod
    def search_after_method(elem_name):
        method_found = []
        after_path = 'after.java'
        liz = lizard.analyze_file(after_path)
        for liz_elem in liz.function_list:
            if liz_elem.long_name == elem_name:
                method_found.append(liz_elem)
                break
        return method_found

    @staticmethod
    def extract_method(liz_elem, before_or_after):
        if before_or_after == 'after':
            file_path = 'after.java'
        else:
            file_path = 'before.java'
            
        code = [line for line in open(file_path)]
        
        method_extracted = code[liz_elem.start_line - 1: liz_elem.end_line]
        
        if method_extracted and method_extracted[-1].strip() and not method_extracted[-1].strip().endswith('}'):
            return []
        
        return method_extracted



    def extract_marked_method(liz_elem, ref, data_type):
        # NOTE: We are using line-level marking for all data sources (GitHub and Gerrit)
        ref = ref[:2]
        m, f = Analyzer1.extract_marked_method_github(liz_elem, ref)
        return m, f


    @staticmethod
    def extract_marked_method_github(liz_elem, ref):
        flag_marked = False
        file_path = 'before.java'
        code = [line for line in open(file_path)]
        method_extracted_marked = []
        for k in range(len(code)):
            if liz_elem.start_line - 1 <= k <= liz_elem.end_line - 1:
                if code[k].endswith('\n'):
                    current_line = code[k][:-1]
                else:
                    current_line = code[k]
                if k + 1 == ref[0]:
                    if ref[0] == ref[1]:
                        current_line = ' START ' + current_line + ' END '
                        flag_marked = True
                    else:
                        current_line = ' START ' + current_line
                if k + 1 == ref[1] and not flag_marked:
                    current_line += ' END '
                    flag_marked = True
                method_extracted_marked.append(current_line)
        return method_extracted_marked, flag_marked


    @staticmethod
    def extract_marked_method_gerrit(liz_elem, ref):
        flag_marked = False
        flag_char = False
        if ref[2] != 0 or ref[3] != 0:
            flag_char = True
        before_path = 'before.java'
        file = open(before_path, 'r')
        code = [line for line in file]
        file.close()
        method_extracted_marked = []
        for k in range(len(code)):
            if liz_elem.start_line - 1 <= k <= liz_elem.end_line - 1:
                if code[k].endswith('\n'):
                    current_line = code[k][:-1]
                else:
                    current_line = code[k]
                if k + 1 == ref[0]:  # comment_start_line
                    if not flag_char:
                        current_line = ' START ' + current_line
                    else:
                        if ref[0] == ref[1]:
                            new_current_line = Analyzer1.add_end(current_line, ref[3])
                            if new_current_line == current_line:
                                break
                            else:
                                current_line = new_current_line
                            new_current_line = Analyzer1.add_start(current_line, ref[2])
                            if new_current_line == current_line:
                                break
                            else:
                                current_line = new_current_line
                                flag_marked = True
                        else:
                            new_current_line = Analyzer1.add_start(current_line, ref[2])
                            if new_current_line == current_line:
                                break
                            else:
                                current_line = new_current_line
                if k + 1 == ref[1] and not flag_marked:
                    if not flag_char:
                        current_line = current_line + ' END '
                        flag_marked = True
                    else:
                        new_current_line = Analyzer1.add_end(current_line, ref[3])
                        if new_current_line == current_line:
                            break
                        else:
                            current_line = new_current_line
                            flag_marked = True
                method_extracted_marked.append(current_line)
        return method_extracted_marked, flag_marked


    @staticmethod
    def read_and_clean_diff(diff_file_path):
        with open(diff_file_path, 'r') as file:
            diff_lines = file.readlines()
    
        # Remove completely empty lines, preserve indentation
        non_empty_lines = [line.rstrip('\n') for line in diff_lines if line.strip()]
    
        clean_diff = '\n'.join(non_empty_lines)
    
        # Write cleaned content back to the same file
        with open(diff_file_path, 'w') as file:
            file.write(clean_diff + '\n')  # Add trailing newline if desired
    
        return clean_diff



        
    @staticmethod
    def detect_comment_lines(lines):
        """
        Detects comment lines in a diff hunk by reconstructing the content first,
        and then marking lines inside block comments (/* ... */), even when incomplete.
    
        Args:
            lines (List[str]): Lines from a unified diff (may begin with '+', '-', or ' ').
    
        Returns:
            List[bool]: True if a line is a comment (including block or line comments).
        """
        # Step 1: Strip diff prefixes and rebuild clean lines
        stripped_lines = [line.lstrip("+- ") for line in lines]
        
        comment_flags = [False] * len(stripped_lines)
        i = 0
        n = len(stripped_lines)
    
        # Step 2: Detect block comments /* ... */
        while i < n:
            line = stripped_lines[i]
            if '/*' in line:
                start_idx = i
                found_closing = False
                for j in range(i, n):
                    if '*/' in stripped_lines[j]:
                        found_closing = True
                        end_idx = j
                        break
                if found_closing:
                    for k in range(start_idx, end_idx + 1):
                        comment_flags[k] = True
                    i = end_idx + 1
                else:
                    # No closing tag, mark all remaining lines
                    for k in range(start_idx, n):
                        comment_flags[k] = True
                    break
            else:
                i += 1
    
        # Step 3: Detect single-line comments (//...) only if not already flagged
        for idx in range(n):
            if not comment_flags[idx] and stripped_lines[idx].strip().startswith("//"):
                comment_flags[idx] = True
    
        return comment_flags
    
    
    

    @staticmethod
    def process_diff_hunk_remove_comments_only(diff_hunk):
        """
        Processes a diff hunk by removing all comment lines.
        
        Args:
            diff_hunk (str): Unified diff hunk as a string.
        
        Returns:
            str: Cleaned diff hunk with all comment lines removed.
        """
        lines = diff_hunk.split("\n")
        comment_flags = Analyzer1.detect_comment_lines(lines)
        comment_removed = any(comment_flags)  # True if any line was marked as a comment
    
        # Keep only non-comment lines
        cleaned_lines = [line for line, is_comment in zip(lines, comment_flags) if not is_comment]
    
        return "\n".join(cleaned_lines), comment_removed

    


    @staticmethod
    def add_start(text, start_char):#only for gerrit 
        try:
            if text[start_char - 1] == ' ':
                new_text = text[:start_char - 1] + ' START ' + text[start_char - 1:]
            else:
                flag = False
                k = 2
                while not flag and start_char - k > 0:
                    if text[start_char - k] == ' ':
                        flag = True
                    else:
                        k += 1
                if flag:
                    new_text = text[:start_char - k] + ' START' + text[start_char - k:]
                else:
                    new_text = ' START ' + text
        except:
            print('WARNING ADD START')
            print('text: ', text)
            print('start_char: ', start_char)
            new_text = text
        return new_text




    @staticmethod
    def add_end(text, end_char):#only for gerrit
        try:
            if text[end_char - 1] == ' ':
                new_text = text[:end_char - 1] + ' END ' + text[end_char - 1:]
            else:
                flag = False
                k = 2
                while not flag and end_char + k < len(text):
                    if text[end_char + k] == ' ':
                        flag = True
                    else:
                        k += 1
                if flag:
                    new_text = text[:end_char + k] + ' END ' + text[end_char + k:]
                else:
                    new_text = text + ' END '
        except:
            print('WARNING ADD END:')
            print('text: ', text)
            print('end_char: ', end_char)
            new_text = text

        return new_text  
    


    @staticmethod
    def strip_tags_from_new_and_check_old(old: str, new: str):
        """
        Checks whether <START> or <END> tokens exist in the old string.
        Removes these tokens from the new string.
        
        Args:
            old (str): Original 'old' string (may include tags)
            new (str): Target 'new' string (will be cleaned of tags)
    
        Returns:
            Tuple[str, bool, bool]:
                - cleaned_new (str): new string with <START> and <END> removed
                - found_start_in_old (bool): True if <START> was in old
                - found_end_in_old (bool): True if <END> was in old
        """
        found_start = ' START ' in old
        found_end = ' END ' in old
    
        cleaned_new = new.replace(' START ', '').replace(' END ', '')
    
        return cleaned_new, found_start, found_end
    
    

    @staticmethod
    def safe_join_lines(lines):
        if isinstance(lines, str):
            return lines
        elif isinstance(lines, list):
            return ''.join(str(line) for line in lines)
        else:
            return str(lines)



    def extraction(self):
        df = self.data
    
        # Method-level parallel lists
        method_Record_id = []
        method_references = []
        method_comment = []
        method_before = []
        method_before_marked = []
        method_after = []
        method_owner_comment = []
        method_comment_to_comment = []
        method_flag_linked = []
        method_creation_time = []
    
        # Diff-level parallel lists
        diff_Record_id = []
        diff_references = []
        diff_comment = []
        diff_old = []
        diff_new = []
        diff_old_wo = []
        diff_new_wo = []
        diff_is_tagged = []
        diff_owner_comment = []
        diff_comment_to_comment = []
        diff_creation_time = []
    
        for i in range(len(df)):
            print("-----------------------------------------------------------------------------------------------------")
            print(f"Processing row: {i}")
    
            try:
                comment = str(df.iloc[i]['message']).strip().lower() if pd.notnull(df.iloc[i]['message']) else ""
    
                if self.data_type == 'Gerrit':
                    if df.iloc[i]['line'] == 0 and df.iloc[i]['comment_start_line'] == 0:
                        self.no_valid_ref += 1
                        continue
    
                id_ref, num_ref, ref, filename = Analyzer1.get_info(df, i, self.data_type)
                ref = [int(r) if isinstance(r, np.float64) else r for r in ref]
    
                Analyzer1.save_temp_code(df, i, self.data_type)
                if not Analyzer1.check_len_code(ref[0], ref[1]):
                    self.no_valid_ref += 1
                    continue
    
                comment_time = df.iloc[i]['comment_updated'] if self.data_type == 'Gerrit' else df.iloc[i]['updated_at']
                record_id = df.iloc[i]['id']
    
                is_comment_to_comment = self.check_comment_to_comment(ref[0], ref[1])
                is_owner_comment = (
                    df.iloc[i]['change_owner'] == 'owner'
                    if self.data_type == 'Gerrit'
                    else df.iloc[i]['user_id'] == df.iloc[i]['owner_id']
                )
                if is_comment_to_comment:
                    self.comm_to_comm += 1
    
                # ----------- Diff Extraction -----------
                diff_flag_marked = Analyzer1.save_marked_diff(ref)
                marked_diff_path = "full_marked_diff_file.diff"
                Marked_clean_diff = Analyzer1.read_and_clean_diff(marked_diff_path)
                Marked_hunk_wo_comments, Marked_hunk_comment_removed = Analyzer1.process_diff_hunk_remove_comments_only(Marked_clean_diff)
    
                Marked_raw_pieces = Analyzer1.split_diff_hunk(Marked_clean_diff)
                Marked_wo_pieces = Analyzer1.split_diff_hunk(Marked_hunk_wo_comments if Marked_hunk_comment_removed else Marked_clean_diff)
    
                Marked_piece_raw = ''
                header = ''
                for h, piece in Marked_raw_pieces.items():
                    if " START " in piece or " END " in piece:
                        Marked_piece_raw = piece
                        header = h
                        break
    
                if Marked_piece_raw and header:
                    Marked_o_raw, Marked_n_raw = Analyzer1.extract_old_file_and_target_from_diff_hunk(Marked_piece_raw)
                    Marked_new_NoTag, _, _ = Analyzer1.strip_tags_from_new_and_check_old(Marked_o_raw, Marked_n_raw)
    
                    Marked_piece_wo = Marked_wo_pieces.get(header, "").strip()
                    if Marked_piece_wo:
                        Marked_o_wo, Marked_n_wo = Analyzer1.extract_old_file_and_target_from_diff_hunk(Marked_piece_wo)
                        Marked_new_wo_NoTag, _, _ = Analyzer1.strip_tags_from_new_and_check_old(Marked_o_wo, Marked_n_wo)
                    else:
                        Marked_o_wo = ''
                        Marked_new_wo_NoTag = ''

   
                    diff_Record_id.append(record_id)
                    diff_references.append(ref)
                    diff_comment.append(comment)
                    diff_old.append(Marked_o_raw)
                    diff_new.append(Marked_new_NoTag)
                    diff_old_wo.append(Marked_o_wo)
                    diff_new_wo.append(Marked_new_wo_NoTag)
                    diff_is_tagged.append(diff_flag_marked)
                    diff_owner_comment.append(is_owner_comment)
                    diff_comment_to_comment.append(is_comment_to_comment)
                    diff_creation_time.append(comment_time)
    
                # ----------- Method Extraction -----------
                before_found = Analyzer1.search_before_method(ref)
                if len(before_found) == 0:
                    self.no_method_before += 1
                    continue
    
                before = Analyzer1.extract_method(before_found[0], 'before')
                if len(before) == 0:
                    self.no_method_before += 1
                    continue
    
                before_marked, flag_marked = Analyzer1.extract_marked_method(before_found[0], ref, self.data_type)
                signature = before_found[0].long_name
                after_found = Analyzer1.search_after_method(signature)
    
                if len(after_found) == 0:
                    self.no_method_after += 1
                    after = ''
                else:
                    after = Analyzer1.extract_method(after_found[0], 'after')
                    if len(after) == 0:
                        after = ''
    
                method_Record_id.append(record_id)
                method_references.append(ref)
                method_comment.append(comment)
                method_before.append(''.join(before))
                method_before_marked.append(''.join(before_marked))
                method_after.append(''.join(after))
                method_owner_comment.append(is_owner_comment)
                method_comment_to_comment.append(is_comment_to_comment)
                method_flag_linked.append(flag_marked)
                method_creation_time.append(comment_time)
    
            except Exception as e:
                print(f"⚠️ An error occurred while processing row index {i}: {e}")
                continue
    
        # Create DataFrames
        method_dfr = pd.DataFrame({
            'Record_id': method_Record_id,
            'references': method_references,
            'comment': method_comment,
            'before': method_before,
            'before_marked': method_before_marked,
            'after': method_after,
            'owner_comment': method_owner_comment,
            'comment_to_comment': method_comment_to_comment,
            'method_flag_linked': method_flag_linked,
            'creation_time': method_creation_time
        })
    
        diff_dfr = pd.DataFrame({
            'Record_id': diff_Record_id,
            'references': diff_references,
            'comment': diff_comment,
            'old': diff_old,
            'new': diff_new,
            'old_wo_comment': diff_old_wo,
            'new_wo_comment': diff_new_wo,
            'Diff_is_tagged': diff_is_tagged,
            'owner_comment': diff_owner_comment,
            'comment_to_comment': diff_comment_to_comment,
            'creation_time': diff_creation_time
        })
    
        # Print summaries
        print("\n📐 Method DataFrame Summary:")
        print("Size:", method_dfr.shape)
        print("Column Data Types:")
        print(method_dfr.dtypes)
        print("\n🔍 First Row (method_dfr):")
        if not method_dfr.empty:
            print(method_dfr.iloc[0])
        else:
            print("❗ method_dfr is empty.")
    
        print("\n📐 Diff DataFrame Summary:")
        print("Size:", diff_dfr.shape)
        print("Column Data Types:")
        print(diff_dfr.dtypes)
        print("\n🔍 First Row (diff_dfr):")
        if not diff_dfr.empty:
            print(diff_dfr.iloc[0])
        else:
            print("❗ diff_dfr is empty.")
    
        print("\n✅ Extraction finished")
        print(f"✔️ Total valid rows processed: {len(df)}")
        print(f"✔️ Method rows created: {len(method_Record_id)}")
        print(f"✔️ Diff rows created: {len(diff_Record_id)}")
        print(f"❌ Skipped due to no valid ref: {self.no_valid_ref}")
        print(f"❌ Skipped due to missing before method: {self.no_method_before}")
        print(f"💬 Skipped due to comment-to-comment: {self.comm_to_comm}")
    
        return method_dfr, diff_dfr

    
