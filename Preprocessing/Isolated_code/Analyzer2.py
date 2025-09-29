import Abstracter
import pandas as pd
import subprocess
from tqdm import tqdm
import re
from langdetect import detect, DetectorFactory
from textblob import TextBlob
import regex
DetectorFactory.seed = 0
from langid.langid import LanguageIdentifier, model  # second language classifier
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
import cld3  # third language classifier
import numpy as np



class Analyzer2:
    """Class to analyze data from Gerrit or GitHub
       The data must be organized in a pandas DataFrame"""

    def __init__(self, data, idioms_path):
        self.data = data
        self.idioms_path = idioms_path
        # info
        self.no_coment = 0
        self.abs_error = 0
        self.comment_abs_error = 0
        self.new_tokens = 0
        self.before_equal_after = 0
        #self.too_long = 0
        self.no_marked_abs = 0
        self.empty_abs_comment = 0
        self.comment_irrelevant = 0
        self.empty_cleaned_comment = 0

    def analyze(self):
        
        record_ids = []
        comments = []
        cleaned_comments = []
        cleaned_comments_wo_abs_ = []
        abstracted_comments = []
        before_ = []
        before_marked_ = []
        before_abs_ = []
        before_abs_marked_ = []
        after_ = []
        after_abs_ = []
    
    
    
        df = self.data
        print("\n📊 Input lengths:")
        print(len(df))
        
        for i in tqdm(range(len(df))):
            try:
            
                before = df.iloc[i]['before']
                before_marked = df.iloc[i]['before_marked']
                after = df.iloc[i]['after']

                
                try:
                    comment = df.iloc[i]['comment']
                except (KeyError, IndexError):
                    print(f"Warning: Missing 'comment' at row {i}, defaulting to empty string.")
                    comment = ""
        
                # Treat NaN or float as empty string
                if pd.isna(comment) or type(comment) == float:
                    comment = ""
     
                # a lot of weird symbols found while processing
                comment = comment.replace("`", "").replace("←", "<-").replace("१२", "").replace("⁻⁵", "").replace("•", "") \
                    .replace("．", "").replace("￼", "").replace("°", "").replace("»«", "").replace("ｔ", "t") \
                    .replace("е", "e").replace("≈", "").replace("⇒", "").replace("¯", "").replace("۸", "").replace("३", "") \
                    .replace("‘", "'").replace("➜", "->").replace("≠", "!=").replace("？", "?").replace("¦¦", "") \
                    .replace("�", "").replace("ह", "").replace("µ", "mu").replace("с", "c").replace("×", "x") \
                    .replace("»", "").replace("²", "").replace("️", "").replace("ö", "").replace("ô", "").replace("ó", "") \
                    .replace("ツ", "").replace("⌘", "").replace("«", "").replace("„", "").replace("·", "").replace("İ", "") \
                    .replace("。。。", "...").replace("λ", "lambda").replace("§", "").replace("ø", "").replace("�", "") \
                    .replace("￼", "").replace("→", "->").replace("´", "'").replace("…", " ").replace("–", "-") \
                    .replace("—", "-").replace("─", "-").replace("’", "'").replace("≤", "<=").replace("≥", ">=") \
                    .replace("∞", "inf").replace("±", "+-").replace("“", "\"").replace("”", "\"").replace("└", " ") \
                    .replace("├", " ").replace("（", "(").replace("）", ")")
    
                
        
                cleaned_comment_wo_abs = Analyzer2.clean_comment_without_abstraction(comment, self.idioms_path)
    
                before = Analyzer2.removeComments(before)
                before_marked = Analyzer2.removeComments(before_marked)
                after = Analyzer2.removeComments(after)
                
                # abstraction
                if Analyzer2.abstraction(before, after, self.idioms_path):
                    if not Analyzer2.is_method():
                        self.abs_error += 1
                        continue
                else:
                    self.abs_error += 1
                    continue
                # add <START>, <END> tokens in before_abs method
                before_marked_lines = before_marked.split('\n')
                flag_mark_abs, before_abs_marked = Analyzer2.mark_abs_method(before_marked_lines, self.idioms_path)
                if not flag_mark_abs:
                    self.no_marked_abs += 1
                    continue
    
    
    
                before_abs, after_abs = Analyzer2.read_abs_files()
                if before_abs is None or after_abs is None:
                    self.abs_error += 1
                    before_abs, after_abs = np.nan, np.nan
                
                comment_abs = Analyzer2.abstract_comment(comment, self.idioms_path)
                if comment_abs is None:
                    self.comment_abs_error += 1
                    comment_abs = np.nan
                else:
                    cleaned_abs_comment = Analyzer2.clean_comment(comment_abs, self.idioms_path)
            
    
                before = Analyzer2.cleanString(before)
                before_marked = Analyzer2.cleanString(before_marked)
                after = Analyzer2.cleanString(after)
    
     
                record_ids.append(df.iloc[i]['Record_id'])
                comments.append(comment)
                cleaned_comments_wo_abs_.append(cleaned_comment_wo_abs)
                cleaned_comments.append(cleaned_abs_comment)
                abstracted_comments.append(comment_abs)
                before_.append(before)
                before_marked_.append(before_marked)
                before_abs_.append(before_abs)
                before_abs_marked_.append(before_abs_marked)
                after_.append(after)
                after_abs_.append(after_abs)
                
                
            except Exception as e:
                print(f"❌ Exception on row {i}: {e}. Skipping row.")
                self.abs_error += 1
                continue



        print("\n✅ Done iterating over rows.")
        print(f"❌ abs_error count: {self.abs_error}")
        print(f"❌ no_marked_abs count: {self.no_marked_abs}")
        print(f"❌ comment_abs_error count: {self.comment_abs_error}")
    
        print("\n📊 Final array lengths:")
        print(" - record_ids:", len(record_ids))
        print(" - comments:", len(comments))
        print(" - cleaned_comments_wo_abs_:", len(cleaned_comments_wo_abs_))
        print(" - cleaned_comments:", len(cleaned_comments))
        print(" - abstracted_comments:", len(abstracted_comments))
        print(" - before_:", len(before_))
        print(" - before_marked_:", len(before_marked_))
        print(" - before_abs_:", len(before_abs_))
        print(" - before_abs_marked_:", len(before_abs_marked_))
        print(" - after_:", len(after_))
        print(" - after_abs_:", len(after_abs_))


        dfr = pd.DataFrame({
            'Record_id': record_ids,
            'comment': comments,
            'cleaned_abstracted_comment': cleaned_comments,
            'cleaned_comment_wo_abs': cleaned_comments_wo_abs_,
            'abstracted_comment': abstracted_comments,
            'before': before_,
            'before_marked': before_marked_,
            'before_abs': before_abs_,
            'before_abs_marked': before_abs_marked_,
            'after': after_,
            'after_abs': after_abs_
        })

        
        return dfr
        
    
    
    
    
    def analyze_wo_abs(self):
        
        record_ids = []
        comments = []
        cleaned_comments_wo_abs_ = []
        before_ = []
        before_marked_ = []
        after_ = []
    

        df = self.data
        print("\n📊 Input lengths:")
        print(len(df))
        
        for i in tqdm(range(len(df))):
            try:
            
                before = df.iloc[i]['before']
                before_marked = df.iloc[i]['before_marked']
                after = df.iloc[i]['after']
                
                try:
                    comment = df.iloc[i]['comment']
                except (KeyError, IndexError):
                    print(f"Warning: Missing 'comment' at row {i}, defaulting to empty string.")
                    comment = ""
        
                # Treat NaN or float as empty string
                if pd.isna(comment) or type(comment) == float:
                    comment = ""
                    
                    
     
                # a lot of weird symbols found while processing
                comment = comment.replace("`", "").replace("←", "<-").replace("१२", "").replace("⁻⁵", "").replace("•", "") \
                    .replace("．", "").replace("￼", "").replace("°", "").replace("»«", "").replace("ｔ", "t") \
                    .replace("е", "e").replace("≈", "").replace("⇒", "").replace("¯", "").replace("۸", "").replace("३", "") \
                    .replace("‘", "'").replace("➜", "->").replace("≠", "!=").replace("？", "?").replace("¦¦", "") \
                    .replace("�", "").replace("ह", "").replace("µ", "mu").replace("с", "c").replace("×", "x") \
                    .replace("»", "").replace("²", "").replace("️", "").replace("ö", "").replace("ô", "").replace("ó", "") \
                    .replace("ツ", "").replace("⌘", "").replace("«", "").replace("„", "").replace("·", "").replace("İ", "") \
                    .replace("。。。", "...").replace("λ", "lambda").replace("§", "").replace("ø", "").replace("�", "") \
                    .replace("￼", "").replace("→", "->").replace("´", "'").replace("…", " ").replace("–", "-") \
                    .replace("—", "-").replace("─", "-").replace("’", "'").replace("≤", "<=").replace("≥", ">=") \
                    .replace("∞", "inf").replace("±", "+-").replace("“", "\"").replace("”", "\"").replace("└", " ") \
                    .replace("├", " ").replace("（", "(").replace("）", ")")
    
                
        
                cleaned_comment_wo_abs = Analyzer2.clean_comment_without_abstraction(comment, self.idioms_path)
    
                before = Analyzer2.removeComments(before)
                before_marked = Analyzer2.removeComments(before_marked)
                after = Analyzer2.removeComments(after)
              
                before = Analyzer2.cleanString(before)
                before_marked = Analyzer2.cleanString(before_marked)
                after = Analyzer2.cleanString(after)
    
     
                record_ids.append(df.iloc[i]['Record_id'])
                comments.append(comment)
                cleaned_comments_wo_abs_.append(cleaned_comment_wo_abs)
                before_.append(before)
                before_marked_.append(before_marked)
                after_.append(after)
                
                
            except Exception as e:
                print(f"❌ Exception on row {i}: {e}. Skipping row.")
                self.abs_error += 1
                continue



        print("\n✅ Done iterating over rows.")
        print(f"❌ abs_error count: {self.abs_error}")
        print(f"❌ no_marked_abs count: {self.no_marked_abs}")
        print(f"❌ comment_abs_error count: {self.comment_abs_error}")
    
        print("\n📊 Final array lengths:")
        print(" - record_ids:", len(record_ids))
        print(" - comments:", len(comments))
        print(" - cleaned_comments_wo_abs_:", len(cleaned_comments_wo_abs_))
        print(" - before_:", len(before_))
        print(" - before_marked_:", len(before_marked_))
        print(" - after_:", len(after_))


        dfr = pd.DataFrame({
            'Record_id': record_ids,
            'comment': comments,
            'cleaned_comment_wo_abs': cleaned_comments_wo_abs_,
            'before': before_,
            'before_marked': before_marked_,
            'after': after_
        })

        
        return dfr
    
    
    

    def ReAbstract(self):
        df = self.data
    
        for i in tqdm(range(len(df))):
            print("-----------------------------------------------------------------------------------------------------")
            print(i)
            try:
                row = df.iloc[i]
                before = row['before']
                before_marked = row['before_marked']
                after = row['after']
                
                try:
                    comment = comment = row['comment']
                except (KeyError, IndexError):
                    print(f"Warning: Missing 'comment' at row {i}, defaulting to empty string.")
                    comment = ""
        
                # Treat NaN or float as empty string
                if pd.isna(comment) or type(comment) == float:
                    comment = ""
    
                comment = Analyzer2.clean_text(comment)
    
                if pd.isna(row['before_abs']) or pd.isna(row['before_abs_marked']) or pd.isna(row['after_abs']) or pd.isna(row['abstracted_comment']):
                    if not Analyzer2.abstraction(before, after, self.idioms_path) or not Analyzer2.is_method():
                        self.abs_error += 1
                        continue
    
                    before_abs, after_abs = Analyzer2.read_abs_files()
                    comment_abs = Analyzer2.abstract_comment(comment, self.idioms_path)
    
                    before_marked_lines = before_marked.split('\n')
                    flag_mark_abs, before_abs_marked = Analyzer2.mark_abs_method(before_marked_lines, self.idioms_path)
                    if not flag_mark_abs:
                        self.no_marked_abs += 1
                        before_abs_marked = row['before_abs_marked']
    
                    # Update DataFrame
                    df.at[i, 'abstracted_comment'] = comment_abs
                    df.at[i, 'before_abs'] = before_abs
                    df.at[i, 'before_abs_marked'] = before_abs_marked
                    df.at[i, 'after_abs'] = after_abs
    
            except Exception as e:
                print(f"❌ Exception on row {i}: {e}. Skipping re-abstraction.")
                self.abs_error += 1
                continue
    
        return df

    
    
    
    
    
    def analyze_diff_level(self):
            
        df = self.data
        
        cleaned_comments = []
        cleaned_comments_no_praise = []
        filter_flags = []
        diffComment_is_relevant = []
        
        
        for i in range(len(df)):
            print("-----------------------------------------------------------------------------------------------------")
            print(i)
        
            
            try:
                comment = df.iloc[i]['comment']
            except (KeyError, IndexError):
                print(f"Warning: Missing 'comment' at row {i}, defaulting to empty string.")
                comment = ""
    
            # Treat NaN or float as empty string
            if pd.isna(comment) or type(comment) == float:
                comment = ""

            
            # a lot of weird symbols found while processing
            comment = comment.replace("`", "").replace("←", "<-").replace("१२", "").replace("⁻⁵", "").replace("•", "") \
                .replace("．", "").replace("￼", "").replace("°", "").replace("»«", "").replace("ｔ", "t") \
                .replace("е", "e").replace("≈", "").replace("⇒", "").replace("¯", "").replace("۸", "").replace("३", "") \
                .replace("‘", "'").replace("➜", "->").replace("≠", "!=").replace("？", "?").replace("¦¦", "") \
                .replace("�", "").replace("ह", "").replace("µ", "mu").replace("с", "c").replace("×", "x") \
                .replace("»", "").replace("²", "").replace("️", "").replace("ö", "").replace("ô", "").replace("ó", "") \
                .replace("ツ", "").replace("⌘", "").replace("«", "").replace("„", "").replace("·", "").replace("İ", "") \
                .replace("。。。", "...").replace("λ", "lambda").replace("§", "").replace("ø", "").replace("�", "") \
                .replace("￼", "").replace("→", "->").replace("´", "'").replace("…", " ").replace("–", "-") \
                .replace("—", "-").replace("─", "-").replace("’", "'").replace("≤", "<=").replace("≥", ">=") \
                .replace("∞", "inf").replace("±", "+-").replace("“", "\"").replace("”", "\"").replace("└", " ") \
                .replace("├", " ").replace("（", "(").replace("）", ")")

            

            # Clean the comment
            cleaned_comment = Analyzer2.clean_comment_without_abstraction(comment, self.idioms_path)
            cleaned_comment_wo_praise = Analyzer2.detect_and_remove_praise(cleaned_comment)
            
            cleaned_comments.append(cleaned_comment)
            cleaned_comments_no_praise.append(cleaned_comment_wo_praise)
            
            # Check if the comment passes the size filter and add the result to the list
            filter_flags.append(Analyzer2.filter_comments_by_length(cleaned_comment_wo_praise))
            
            
            if Analyzer2.isCommentRelevant(cleaned_comment_wo_praise, self.idioms_path):
                diffComment_is_relevant.append(True)
            else:
                diffComment_is_relevant.append(False)

        
        # Add the two new columns to the DataFrame
        df['cleaned_comment'] = cleaned_comments
        df['cleaned_comment_no_praise'] = cleaned_comments_no_praise
        df['diff_Comment_is_relevant'] = diffComment_is_relevant
        df['Comment_size_filter_3_150'] = filter_flags

        return df

    
    def removeExtraSpaces(s):
        return " ".join(s.split())

    @staticmethod
    def cleanString(s):
        s = s.replace("\n", ' ').replace("\t", ' ').replace("</s>", ' ').replace("<s>", ' ').strip()
        return Analyzer2.removeExtraSpaces(s)
    
    
    
    def drop_rows_with_nan(self):
        df = self.data
        columns_to_check = ['before_abs', 'after_abs', 'before_abs_marked', 'abstracted_comment']
        df_dropped = df.dropna(subset=columns_to_check, how='any').reset_index(drop=True)
        return df_dropped
    
    
    def drop_rows_with_nan_diff(self):
        df = self.data
        columns_to_check = ['cleaned_comment', 'cleaned_comment_no_praise']
        df_dropped = df.dropna(subset=columns_to_check, how='any').reset_index(drop=True)
        return df_dropped
    

    def info_for_cleaning_paper1_isolated(self):

        dfr = self.data

        comment_is_relevant = []
        clean_comment_empty=[]
        new_token_in_after = []
        before_equal_after = []
        key_words = ['INT', 'FLOAT', 'ANNOTATION', 'STRING', 'CHAR', 'METHOD', 'TYPE', 'VAR']
        

        for i in range(len(dfr)):
            
            print("info_for_cleaning_paper1_isolated-----------------------------------------------------------------------------------------------------")
            print(i)
            
            Record_ID = dfr.iloc[i]['Record_id']
            before = dfr.iloc[i]['before']
            before = dfr.iloc[i]['before_marked']
            after = dfr.iloc[i]['after']
            
            
            
            try:
                comment = dfr.iloc[i]['comment']
            except (KeyError, IndexError):
                print(f"Warning: Missing 'comment' at row {i}, defaulting to empty string.")
                comment = ""
    
            # Treat NaN or float as empty string
            if pd.isna(comment) or type(comment) == float:
                comment = ""
            

            # a lot of weird symbols found while processing
            comment = comment.replace("`", "").replace("←", "<-").replace("१२", "").replace("⁻⁵", "").replace("•", "") \
                .replace("．", "").replace("￼", "").replace("°", "").replace("»«", "").replace("ｔ", "t") \
                .replace("е", "e").replace("≈", "").replace("⇒", "").replace("¯", "").replace("۸", "").replace("३", "") \
                .replace("‘", "'").replace("➜", "->").replace("≠", "!=").replace("？", "?").replace("¦¦", "") \
                .replace("�", "").replace("ह", "").replace("µ", "mu").replace("с", "c").replace("×", "x") \
                .replace("»", "").replace("²", "").replace("️", "").replace("ö", "").replace("ô", "").replace("ó", "") \
                .replace("ツ", "").replace("⌘", "").replace("«", "").replace("„", "").replace("·", "").replace("İ", "") \
                .replace("。。。", "...").replace("λ", "lambda").replace("§", "").replace("ø", "").replace("�", "") \
                .replace("￼", "").replace("→", "->").replace("´", "'").replace("…", " ").replace("–", "-") \
                .replace("—", "-").replace("─", "-").replace("’", "'").replace("≤", "<=").replace("≥", ">=") \
                .replace("∞", "inf").replace("±", "+-").replace("“", "\"").replace("”", "\"").replace("└", " ") \
                .replace("├", " ").replace("（", "(").replace("）", ")")

            cleaned_comment = Analyzer2.clean_comment_without_abstraction(comment, self.idioms_path)

            new_token = False  # Variable to hold whether condition is met for each row
            before_tokens = set(before.split())
            tokens = set(after.split())
            for token in tokens:
                if any(token.startswith(key_word) for key_word in key_words) and token not in before_tokens:                    
                    new_token = True  # Update the variable if condition is met
    
            new_token_in_after.append(new_token)    
                
            
            
            
            if cleaned_comment == '':
                clean_comment_empty.append(True)
                
            else:
                clean_comment_empty.append(False)
        
        
            if Analyzer2.isCommentRelevant(cleaned_comment, self.idioms_path):
                comment_is_relevant.append(True)
            else:
                comment_is_relevant.append(False)
                
        
        self.data['new_token_in_after'] = new_token_in_after
        self.data['clean_comment_empty'] = clean_comment_empty
        self.data['comment_is_relevant'] = comment_is_relevant
        
        
        
        
        
    def info_for_cleaning_paper1(self):
        
        dfr = self.data
        
        abs_comment_is_relevant = []
        wo_abs_comment_is_relevant = []
        abs_comment_is_empty = []
        clean_comment_wo_abs_empty=[]
        new_token_in_abs_after = []
        abs_before_equal_abs_after = []
        
        key_words = ['INT', 'FLOAT', 'ANNOTATION', 'STRING', 'CHAR', 'METHOD', 'TYPE', 'VAR']
        

        for i in range(len(dfr)):
            
            print("info_for_cleaning_paper1-----------------------------------------------------------------------------------------------------")
            print(i)
            
            
            Record_ID = dfr.iloc[i]['Record_id']
            
            before = dfr.iloc[i]['before']
            before_abs = dfr.iloc[i]['before_abs']
            beforeAbs_marked =  dfr.iloc[i]['before_abs_marked']
            
            
            after = dfr.iloc[i]['after']
            after_abs = dfr.iloc[i]['after_abs']
            
            
            
            clean_comment_wo_abs = dfr.iloc[i]['cleaned_comment_wo_abs']
            
            abs_comment = dfr.iloc[i]['abstracted_comment']
            clean_abs_comment = dfr.iloc[i]['cleaned_abstracted_comment']
            
            
            try:
                comment = dfr.iloc[i]['comment']
            except (KeyError, IndexError):
                print(f"Warning: Missing 'comment' at row {i}, defaulting to empty string.")
                comment = ""
    
            # Treat NaN or float as empty string
            if pd.isna(comment) or type(comment) == float:
                comment = ""
            
            if before_abs == after_abs:
                abs_before_equal_abs_after.append(True)
            else:
                abs_before_equal_abs_after.append(False)
                

            new_token = False  # Variable to hold whether condition is met for each row
            before_tokens = set(before_abs.split())
            tokens = set(after_abs.split())
            for token in tokens:
                if any(token.startswith(key_word) for key_word in key_words) and token not in before_tokens:                    
                    new_token = True  # Update the variable if condition is met
                    
                

            new_token_in_abs_after.append(new_token)    
                
            
            
            
            if clean_comment_wo_abs == '':
                clean_comment_wo_abs_empty.append(True)
                
            else:
                clean_comment_wo_abs_empty.append(False)
        
        
            if Analyzer2.isCommentRelevant(clean_comment_wo_abs, self.idioms_path):
                wo_abs_comment_is_relevant.append(True)
            else:
                wo_abs_comment_is_relevant.append(False)
                
                
            if clean_abs_comment == '' or str(abs_comment).strip() == '':
                abs_comment_is_empty.append(True)
            else:
                abs_comment_is_empty.append(False)
            
        

            if Analyzer2.comment_is_relevant(beforeAbs_marked, before, comment, abs_comment):
                abs_comment_is_relevant.append(True)
                
            else:
                abs_comment_is_relevant.append(False)
                
                
        self.data['abs_before_equal_abs_after'] = abs_before_equal_abs_after
        self.data['new_token_in_abs_after'] = new_token_in_abs_after
        self.data['clean_comment_wo_abs_empty'] = clean_comment_wo_abs_empty
        self.data['wo_abs_comment_is_relevant'] = wo_abs_comment_is_relevant
        self.data['abs_comment_is_empty'] = abs_comment_is_empty
        self.data['abs_comment_is_relevant'] = abs_comment_is_relevant
                    
        


    def info_for_cleaning_paper2(self):
        
        dfr = self.data
        
        
        wo_abs_comment_is_relevant = []
        clean_comment_wo_abs_empty=[]
        
        

        for i in range(len(dfr)):
            
            print("info_for_cleaning_paper2-----------------------------------------------------------------------------------------------------")
            print(i)
            
        
            try:
                comment = dfr.iloc[i]['comment']
            except (KeyError, IndexError):
                print(f"Warning: Missing 'comment' at row {i}, defaulting to empty string.")
                comment = ""
    
            # Treat NaN or float as empty string
            if pd.isna(comment) or type(comment) == float:
                comment = ""
                
                
                
            clean_comment_wo_abs = dfr.iloc[i]['cleaned_comment_wo_abs']
            

            if clean_comment_wo_abs == '':
                clean_comment_wo_abs_empty.append(True)
                
            else:
                clean_comment_wo_abs_empty.append(False)
        
        
            if Analyzer2.isCommentRelevant(clean_comment_wo_abs, self.idioms_path):
                wo_abs_comment_is_relevant.append(True)
            else:
                wo_abs_comment_is_relevant.append(False)
                
                
                
                
        self.data['clean_comment_wo_abs_empty'] = clean_comment_wo_abs_empty
        self.data['wo_abs_comment_is_relevant'] = wo_abs_comment_is_relevant
        
        
        
    def size_filter(self):
        dfr = self.data
        max_tokens = 100
        rows_to_drop_size = []
        
        
        for i in range(len(dfr)):
            
            before_abs = dfr.iloc[i]['before_abs']
            after_abs = dfr.iloc[i]['after_abs']
            
            if len(before_abs.split()) > max_tokens or len(after_abs.split()) > max_tokens:
                rows_to_drop_size.append(i)
                
                
        if rows_to_drop_size:
            dfr.drop(rows_to_drop_size, inplace=True, errors='ignore')
            dfr.reset_index(drop=True, inplace=True)
            
        return dfr

        
    @staticmethod
    def diff_replace(s1, s2):
        
        s1_words = set(s1.split())
        s2_words = set(s2.split())
        
        # Find words that are in s2 but not in s1
        diff_words = s2_words - s1_words
        diff_words = list(diff_words)
        diff_words = [item for item in diff_words if item not in ['<', '>']]
        
        # Replace the first difference with <START> in s1
        if len(diff_words) == 2:
            s2 = s2.replace(diff_words[0], ' START ', 1)
        
            # Replace the second difference with <END> in s1
    
            s2 = s2.replace(diff_words[1], ' END ', 1)
            
        else:
            s2 = ''
    
        return s2
            

    @staticmethod 
    def extract_records(diff_hunk):
        lines = diff_hunk.strip().split('\n')
        content_start = [i for i, line in enumerate(lines) if line.startswith('@@')]

        records = []
        for i in range(len(content_start)):
            start = content_start[i] + 1
            end = content_start[i + 1] if i + 1 < len(content_start) else len(lines)
            record = lines[start:end]
            hunk_info = lines[content_start[i]].split()[1:3]  # Extracting the @@ markers info
            records.append((hunk_info, record))

        return records
    
    
    @staticmethod 
    def split_versions(record_data):
        hunk_info, record = record_data
        before = []
        after = []
        plus_lines = []

        for line in record:
            if line.startswith('-'):
                before.append(line[1:])
            elif line.startswith('+'):
                after.append(line[1:])
                plus_lines.append(line[1:])
            else:
                before.append(line)
                after.append(line)

        return hunk_info, before, after, plus_lines
    
    

    @staticmethod
    def abstraction(before, after, path_idioms):
        Analyzer2.save_method(before, './utils/before.txt')
        Analyzer2.save_method(after, './utils/after.txt')
        abstracter = Abstracter.Abstracter(path_idioms)
        path_before = './utils/before.txt'
        path_after = './utils/after.txt'
        path_before_abs = './utils/before_abs'
        path_after_abs = './utils/after_abs'
        if abstracter.abstract(path_before, path_after, path_before_abs, path_after_abs):
            return True
        return False


    @staticmethod
    def save_method(method, name):
        f = open(name, 'w')
        f.write(method)
        f.close()

    @staticmethod
    def read_abs_files():
        f = open('./utils/before_abs', 'r')
        before = f.read()
        f.close()
        f = open('./utils/after_abs', 'r')
        after = f.read()
        f.close()
        return before.strip(), after.strip()

    @staticmethod
    def is_method():
        before, after = Analyzer2.read_abs_files()
        if len(before) == 0 or len(after) == 0:
            return False
        if not before.endswith('}') or not after.endswith('}'):
            return False
        return True

    @staticmethod
    def is_before_equals_after():
        before, after = Analyzer2.read_abs_files()
        if before == after:
            return True
        return False

    @staticmethod
    def is_too_long():
        max_tokens = 100
        before, after = Analyzer2.read_abs_files()
        if len(before.split()) > max_tokens or len(after.split()) > max_tokens:
            return True
        return False

    @staticmethod
    def check_new_tokens():
        before, after = Analyzer2.read_abs_files()
        before_tokens = set(before.split())
        dictionary = [line for line in open('./utils/dictionary.txt')]
        tokens = [elem.split(' : ')[0] for elem in dictionary]
        for token in tokens:
            if token not in before_tokens:
                return True
        return False

    @staticmethod
    def mark_abs_method(before_marked, idioms_path):
        official_keywords, unofficial_keywords, idioms = Analyzer2.load_keywords_idioms(idioms_path)
        tokens, values = Analyzer2.load_map()
        word_for_token = Analyzer2.select_word_tokens(idioms, values)
        word_for_special_token = Analyzer2.select_word_special_token(
            idioms, values, word_for_token)
        all_tokens, all_values, max_len = Analyzer2.store_all_tokens(
            official_keywords, unofficial_keywords, idioms,
            values, word_for_token, word_for_special_token)

        tokens, values = Analyzer2.sort_tokens(all_tokens, all_values, max_len)

        f = open('./utils/before_abs', 'r')
        before_abs = f.read()
        f.close()
        m_abs_tokens = before_abs.split()
        m_mark_tokens = Analyzer2.tokenize_method(before_marked, tokens, values, word_for_special_token)
        if len(m_mark_tokens) != len(m_abs_tokens) + 2:
            return False, ''
        else:
            marked_abs = Analyzer2.add_start_end_abs(m_mark_tokens, m_abs_tokens, word_for_special_token)
            if marked_abs == '':
                return False, ''
            return True, marked_abs

    @staticmethod
    def load_keywords_idioms(idioms_path):
        official_keywords = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
                             'char', 'class', 'const', 'continue', 'default', 'do', 'double',
                             'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'if',
                             'goto', 'implements', 'import', 'instanceof', 'int', 'interface',
                             'long', 'native', 'new', 'package', 'private', 'protected', 'public',
                             'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized',
                             'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while']
        unofficial_keywords = ['true', 'false', 'null', '+=', '&&', '-=', '!=', '++', '||',
                               '<=', '>=', '<<=', '>>=', '...', '--', '/=', '>>>=', '<<<=']
        idioms = [line.strip() for line in open(idioms_path)]
        return official_keywords, unofficial_keywords, idioms

    @staticmethod
    def load_map():
        rows = [line.strip() for line in open('./utils/dictionary.txt')]
        keys = []
        values = []
        for row in rows:
            splitted_row = row.split(' : ')
            values.append(splitted_row[0])
            if len(splitted_row) == 2:
                keys.append(splitted_row[1])
            else:
                key = ''
                for elem in splitted_row[1:-1]:
                    key += elem + ' : '
                key += splitted_row[-1]
                keys.append(key)
        return values, keys

    @staticmethod
    def select_word_tokens(idioms, values):
        word_for_token = 'TKN'
        problematic_tokens = ['T', 'K', 'N', 'TK', 'KN', 'TKN']
        for idiom in idioms:
            if idiom in problematic_tokens:
                word_for_token = word_for_token.replace(idiom, '')

        for value in values:
            if value in problematic_tokens:
                word_for_token = word_for_token.replace(value, '')

        if len(word_for_token) == 0:
            # pick the firs available char in the list
            char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            flag = True
            i = 0
            current_char = ''
            while flag:
                current_char = char_list[i]
                if current_char in idioms or current_char in values:
                    i += 1
                else:
                    flag = False
            word_for_token = current_char
        return word_for_token

    @staticmethod
    def select_word_special_token(idioms, values, word_for_token):
        # for SPECIAL TOKENS (START, END)
        word_for_special_token = 'STKN'
        problematic_tokens = ['S', 'T', 'K', 'N', 'ST', 'TK', 'KN', 'STK', 'TKN', 'STKN']
        for idiom in idioms:
            if idiom in problematic_tokens:
                word_for_special_token = word_for_special_token.replace(idiom, '')

        for value in values:
            if value in problematic_tokens:
                word_for_special_token = word_for_special_token.replace(value, '')

        if len(word_for_special_token) == 0:
            # pick the firs available char in the list
            char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            flag = True
            i = 0
            current_char = ''
            while flag:
                current_char = char_list[i]
                if current_char in idioms or current_char in values or current_char == word_for_token:
                    i += 1
                else:
                    flag = False
            word_for_special_token = current_char
        return word_for_special_token

    @staticmethod
    def add_to_all_tokens(elements, all_tokens, all_values, word, max_len):
        for i in range(len(elements)):
            all_tokens.append(word)
            all_values.append((elements[i]))
            if len(elements[i]) > max_len:
                max_len = len(elements[i])
        return all_tokens, all_values, max_len

    @staticmethod
    def store_all_tokens(official_keywords, unofficial_keywords, idioms,
                         values, word_for_token, word_for_special_token):
        all_tokens = []
        all_values = []
        max_len = 0
        # add official keywords
        all_tokens, all_values, max_len = Analyzer2.add_to_all_tokens(
            official_keywords, all_tokens, all_values, word_for_token, max_len)
        # add unofficial keywords
        all_tokens, all_values, max_len = Analyzer2.add_to_all_tokens(
            unofficial_keywords, all_tokens, all_values, word_for_token, max_len)
        # add idioms
        all_tokens, all_values, max_len = Analyzer2.add_to_all_tokens(
            idioms, all_tokens, all_values, word_for_token, max_len)
        # add real tokens/values
        all_tokens, all_values, max_len = Analyzer2.add_to_all_tokens(
            values, all_tokens, all_values, word_for_token, max_len)
        # add specials
        all_tokens.append(word_for_special_token)
        all_values.append(' START ')
        all_tokens.append(word_for_special_token)
        all_values.append(' END ')
        return all_tokens, all_values, max_len

    @staticmethod
    def sort_tokens(all_tokens, all_values, max_len):
        sorted_tokens = []
        sorted_values = []
        i = max_len
        while i > 0:
            for j in range(len(all_tokens)):
                if len(all_values[j]) == i:
                    sorted_values.append(all_values[j])
                    sorted_tokens.append(all_tokens[j])
            i -= 1
        return sorted_tokens, sorted_values

    @staticmethod
    def tokenize_method(m, tokens, values, word_for_special_tokens):
        text = ''
        flag_comment = False
        for i in range(len(m)):
            line = m[i].strip()
            if line == '':
                continue
            if flag_comment:
                if '*/' in line:
                    flag_comment = False
                    if len(line[line.index('*/') + 2:]) > 0:
                        line = line[line.index('*/') + 2:]
                    else:
                        continue
            if line.startswith('/*') and '*/' not in line:
                flag_comment = True
                continue
            for j in range(len(tokens)):
                line = line.replace(values[j], ' ' + tokens[j] + ' ')
            if '//' in line:
                index = line.index('//')
                new_line = line[:index]
                if word_for_special_tokens in line[index:].split():
                    new_line += ' ' + word_for_special_tokens + ' '
                    comm_line = line[index:]
                    index2 = comm_line.index(word_for_special_tokens)
                    if word_for_special_tokens in comm_line[index2 + 2:].split():
                        new_line += word_for_special_tokens + ' '
                line = new_line
            if '/*' in line and '*/' in line:
                flag_comment2 = True
                while flag_comment2:
                    try:
                        line = line[:line.index('/*')] + ' ' + line[line.index('*/') + 2:]
                    except:
                        continue
                    #line = line[:line.index('/*')] + ' ' + line[line.index('*/') + 2:]
                    if '/*' not in line and '*/' not in line:
                        flag_comment2 = False
            text += line + ' '

        special_char = ['(', ')', '[', ']', '{', '}', ';', '>',
                        '.', '<', '-', '!', ',', '/', '~', ':']
        m_tokens = ''.join(char if char not in special_char else " " + char + " " for char in text).split()
        return m_tokens

    @staticmethod
    def add_start_end_abs(m_tokens, m_abs_tokens, special_word):
        flag = True
        m = ''
        j = 0
        for i in range(len(m_tokens)):
            if m_tokens[i] == special_word:
                if flag:
                    m += '<START> '
                    flag = False
                else:
                    m += '<END> '
            else:
                try:
                    m += m_abs_tokens[j] + ' '
                    j += 1
                except:
                    return ''
        return m

    @staticmethod
    def abstract_comment(comm, idioms_path):
        if type(comm) == float:
            return ''
        comm = comm.replace('`', ' ')
        comm = comm.replace('\r', '\n')
        comm_lines = comm.split('\n')
        text = ''
        for i in range(len(comm_lines)):
            text += comm_lines[i].strip()
            values, keys = Analyzer2.load_map()
        # replace strings and char first
        for i in range(len(values)):
            if values[i].startswith('STRING'):
                text = text.replace(keys[i], values[i])
            elif values[i].startswith('CHAR'):
                text = text.replace(keys[i], values[i])
        # remove links
        if text.__contains__('http'):
            text = ''.join(char if char != '(' and char != ')'
                           else " " + char + " " for char in text)
            new_text = ''.join(e + ' ' if not e.__contains__('http') else ' ' for e in text.split())
            text = new_text
        # replace float (to avoid problems with punctuation)
        for i in range(len(values)):
            if values[i].startswith('FLOAT'):
                text.replace(keys[i], values[i])
        # split text
        special_char = ['*', '(', ')', '[', ']', '{', '}', ';', '>',
                        '.', '<', '-', ',', '/', '~', ':', '=', '+',
                        '"', "'", '!', '?']

        comm_tokens = ''.join(char if char not in special_char else " " + char + " " for char in text).split()
        # replace the rest and reconstruct abstracted comment
        comment_abstracted = ''
        for token in comm_tokens:
            if token in keys:
                index = keys.index(token)
                token = values[index]
            else:
                if Analyzer2.is_camel_case(token) and not Analyzer2.is_idiom(token, idioms_path):
                    token = '_CODE_'
            comment_abstracted += token + ' '
        return comment_abstracted

    @staticmethod
    def snippet(c):
        index1 = c.index('<START>')
        index2 = c.index('<END>')
        return c[index1 + 7: index2]

    @staticmethod
    def is_camel_case(s):
        if s != s.lower() and s != s.upper() and "_" not in s:
            s = s[0].lower() + s[1:]
            if s != s.lower():
                return True
        return False

    @staticmethod
    def is_idiom(s, idioms_path):
        idioms = [line.strip() for line in open(idioms_path)]
        return s in idioms

    @staticmethod
    def create_csv_relevant_checking(before, comment, before_abs_marked, snippet, comment_abs):
        dfr = pd.DataFrame({
            'id': [0],
            'code': [before],
            'comment': [comment],
            'is_relevant': [None],
            'refers_previous': [None],
            'code_abs': [before_abs_marked],
            'snippet_abs': [snippet],
            'comment_abs': [comment_abs]
        })

        dfr.to_csv('./utils/comment.csv')

    @staticmethod
    def comment_is_relevant(before_abs_marked, before, comment, comment_abs):
        snippet = Analyzer2.snippet(before_abs_marked)
        Analyzer2.create_csv_relevant_checking(before, comment, before_abs_marked, snippet, comment_abs)
        f = open('./utils/is_relevant.sh', 'w')
        f.close()
        subprocess.run('chmod a+x ./utils/is_relevant.sh', shell=True)
        f = open('./utils/is_relevant.sh', 'w')
        f.write('#!/usr/bin/env bash\n')
        path_jar = './utils/classifyRelevantComments.jar '
        path_stopwords = './utils/stop-words-english.txt '
        path_result = './utils/ '
        path_csv_comment = './utils/comment.csv '
        f.write('java -jar ' + path_jar + path_result + path_csv_comment + path_stopwords + '100')
        f.close()
        subprocess.run('./utils/is_relevant.sh')
        f = open('./utils/is_relevant.txt', 'r')
        is_relevant = f.read()
        f.close()
        if is_relevant.strip() == 'true':
            return True
        return False
    
    @staticmethod
    def removeComments(s):
        block_comment_pattern = regex.compile(r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/")
        inline_marked_comment_pattern = regex.compile(r"(?<!:)\/\/.*?(?=END)")
        inline_comment_pattern = regex.compile(r"(?<!:)\/\/.*")

        s = block_comment_pattern.sub(r' ', s)
        s = inline_marked_comment_pattern.sub(r' ', s)
        s = inline_comment_pattern.sub(r' ', s)

        return s

    @staticmethod
    #for paper2
    def isCommentRelevant(comment, idioms_path):

        comment = str(comment).lower()
        size = len(comment.split())

        if size == 0:
            return False

        if len(Analyzer2.remove_stopwords(comment, idioms_path)) == 0:
            return False

        if comment == ":+1:" or comment == "+1" or \
                comment == "\+" or \
                comment == ":100:" or comment == "..." or \
                comment == "!!!" or comment == "==>" or \
                comment == "++" or \
                comment == "??" or comment == "-;" or \
                comment == "... :)" or comment == "+" or \
                comment == ";)" or comment == ":0" or \
                comment == ":-)" or comment == ":0" or \
                comment == ";-)" or comment == ":(" or \
                comment == ":-(" or comment == "?!" or \
                comment == "^^" or comment == "^^^" or \
                comment == "???" or comment == ":/" or \
                comment == "+:100:" or comment == "????" or \
                comment == "..?" or comment == ":-|" or \
                comment == "...?" or comment == "??????" or \
                comment == ":)" or comment == "^":
            return False

        # Useless comments, one word, no action required or unclear action
        if size == 1:
            if "done" in comment or "idem" in comment or \
                    "lgtm" in comment or "docs" in comment or \
                    "ok" in comment or "nice" in comment or \
                    "pleas" in comment or "ditto" in comment or \
                    "thank" in comment or "lol" in comment or \
                    "fine" in comment or "agre" in comment or \
                    "dito" in comment or "yeh" in comment or \
                    "cool" in comment or "same" in comment or \
                    "ack" in comment or "hahaha" in comment:
                return False

        if size == 2:
            if "ack" in comment or \
                    "same change" in comment or "this too" in comment or \
                    "java doc" in comment or \
                    "good catch" in comment or "and this" in comment:
                return False

        # as above, see above, ditto above, same above,
        # same here, and here, also here, here too, ditto here, here..., likewise here.
        if size <= 3:
            if "here" in comment or "above" in comment:
                return False;

        # Request to change formatting, no impact on code
        if "indent" in comment and size < 5:
            return False

        # Likely a thank you message
        if ("works for me" in comment or "sounds good" in comment or "makes sense" in comment or "smile" in comment
            or "approv" in comment) and size < 5:
            return False

        # Request to add test code, no impact on the reviewed code
        if ("test" in comment and size < 5) or ("add" in comment and "test" in comment):
            return False

        # Request for clarification
        if (("please explain" in comment or "what" in comment or "wat" in comment or "explan" in comment) and size < 5) \
                or ("not sure" in comment and ("understand" in comment or "meant" in comment)):
            return False

        # Refers to previous comment or external resource with unclear action point
        if ("same as" in comment or "same remark" in comment or "said above" in comment or "do the same" in comment) \
                and size < 5:
            return False

        if ("like" in comment or "see" in comment) and ("http" in comment or "https" in comment or "<link_" in comment):
            return False

        # Request to add comment
        if "document" in comment or "javadoc" in comment or "comment" in comment:
            return False

        # Feedback about reorganizing the PR
        if "pr" in comment and size < 5:
            return False

        # Comment contains a +1 to support previous comment.
        # It may be accompanied by another word, like agree or a smile.
        # This is the reason for < 3
        if "+1" in comment and size < 3:
            return False

        # The code is ok for now
        if "for now" in comment and size < 5:
            return False

        # Answers
        if ("fixed" in comment or "thank" in comment or "youre right" in comment) and size < 3:
            return False

        return True

    @staticmethod
    def clean_comment(comm, idioms_path):
        if type(comm) == float:
            return ''
        comm = comm.replace('`', ' ')
        comm = comm.replace('\r', '\n')
        comm_lines = comm.split('\n')
        text = ''
        for i in range(len(comm_lines)):
            text += comm_lines[i].strip()
            values, keys = Analyzer2.load_map()
        # replace strings and char first
        for i in range(len(values)):
            if values[i].startswith('STRING'):
                text = text.replace(keys[i], values[i])
            elif values[i].startswith('CHAR'):
                text = text.replace(keys[i], values[i])
        # remove links
        if text.__contains__('http'):
            text = Analyzer2.remove_http(text)
        # replace float (to avoid problems with punctuation)
        for i in range(len(values)):
            if values[i].startswith('FLOAT'):
                text.replace(keys[i], values[i])

        comm_tokens = Analyzer2.remove_split_punctuation(text).split()
        # replace the rest and reconstruct abstracted comment
        comment_abstracted = ''
        for token in comm_tokens:
            if token in keys:
                index = keys.index(token)
                token = values[index]
            elif token.startswith('@') and not Analyzer2.is_idiom(token, idioms_path):
                token = ''
            else:
                if Analyzer2.is_camel_case(token) and not Analyzer2.is_idiom(token, idioms_path):
                    token = '_CODE_'
            comment_abstracted += token + ' '
        text = Analyzer2.remove_stopwords(comment_abstracted, idioms_path)
        if len(text) == 0:
            return ''
        if text[-1] == '.' or text[-1] == '?' or text[-1] == '!':
            text = text[:-1]
        if len(text) == 0:
            return ''
        text = Analyzer2.lower_text(text, idioms_path)
        return text.strip()



    def praise_phrases_list ():
        return [
        "great job", "well done", "excellent", "good work", "nice work", "fantastic", "amazing",
        "bravo", "keep it up", "you're doing great", "outstanding", "you rock", "awesome", "impressive",
        "LGTM", "Well Done", "Nice Work", "Great Job", "Excellent", "Fantastic", "Awesome", 
        "Impressive", "Clean Code", "Good to Go", "Superb", "Brilliant", "High Quality", "Keep It Up",
        "Amazing Job"]
    
    
    
    def is_praise(comment):
        praise_phrases = Analyzer2.praise_phrases_list ()
        analysis = TextBlob(comment)
        comment_lower = comment.lower()
        keyword_match = any(keyword in comment_lower for keyword in praise_phrases)
        return analysis.sentiment.polarity > 0.5 or keyword_match




    def remove_praise_phrases(input_string):
        praise_phrases = Analyzer2.praise_phrases_list ()
        praise_pattern = r'\b(?:' + '|'.join(re.escape(phrase) for phrase in praise_phrases) + r')\b'
        cleaned_string = re.sub(praise_pattern, '', input_string, flags=re.IGNORECASE)
        cleaned_string = ' '.join(cleaned_string.split()).strip()
        return cleaned_string if cleaned_string else ""


    @staticmethod
    def detect_and_remove_praise(input_string):
        if Analyzer2.is_praise(input_string):
            return Analyzer2.remove_praise_phrases(input_string)
        else:
            return input_string




    @staticmethod
    def clean_comment_without_abstraction(comm, idioms_path): 
        
        if type(comm) == float:
            return ''
        
        # Initial cleaning using clean_text
        comm = Analyzer2.clean_text(comm)
        
        # Check if the comment is in English
        if not Analyzer2.is_english(comm):
            return ''
        
        # Combine the comment lines into a single string
        comm_lines = comm.split('\n')
        text = ' '.join(line.strip() for line in comm_lines)
        
        # Remove HTTP links
        if 'http' in text:
            text = Analyzer2.remove_http(text)
        
        comm_tokens = Analyzer2.remove_split_punctuation_with_floats(text).split()
        
        # Reconstruct the comment
        comment_ = ' '.join(
            token if not (token.startswith('@')) else ''
            for token in comm_tokens
        )
        
        # Remove stopwords
        text = Analyzer2.remove_stopwords(comment_, idioms_path)
        
        
        # Remove trailing punctuation
        if text and text[-1] in '.?!':
            text = text[:-1]
        
        # Lowercase the text
        text = Analyzer2.lower_text(text, idioms_path)
        
        return text.strip()
    
    

    @staticmethod
    def remove_http(text):
        text = ''.join(char if char != '(' and char != ')'
                       else " " + char + " " for char in text)
        new_text = ''.join(e + ' ' if not e.__contains__('http') else ' ' for e in text.split())
        return new_text

    @staticmethod
    def remove_split_punctuation(text):
        # remove some useless char
        remove_char = ['.', ',', '"', "'", '~']
        text = ''.join(char if char not in remove_char else " " for char in text)
        # split text
        special_char = ['*', '(', ')', '[', ']', '{', '}', ';', '>',
                        '<', '-', '/', ':', '=', '+',
                        '"', "'", '!', '?']
        text = ''.join(char if char not in special_char else " " + char + " " for char in text)
        return text

    @staticmethod
    def remove_stopwords(text, idioms_path):
        path = './utils/stop-words-english.txt'
        stopwords = [line.strip() for line in open(path)]
        stopwords.append('nit')
        idioms = [line.strip() for line in open(idioms_path)]
        keywords = Analyzer2.load_keywords()
        # text = ''.join(w + ' ' for w in text.split() if w.lower() not in stopwords).strip()
        text = ''.join(w + ' ' for w in text.split() if not (
                w.lower() in stopwords and w not in idioms and w not in keywords)).strip()
        if len(text) == 0:
            return ''
        punct = [':', '-', '>']
        if text[0] in punct:
            text = text[1:]
        text = text.replace('( : )', '')
        return text
    
    
    


    @staticmethod
    def lower_text(text, idioms_path):
        idioms = [line.strip() for line in open(idioms_path)]
        keywords = Analyzer2.load_keywords()
        new_text = ''
        for token in text.split():
            if token == token.upper() or token in idioms or token in keywords:
                new_text += token + ' '
            else:
                new_text += token.lower() + ' '
        return new_text.strip()

    @staticmethod
    def load_keywords():
        return ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
                'char', 'class', 'const', 'continue', 'default', 'do', 'double',
                'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'if',
                'goto', 'implements', 'import', 'instanceof', 'int', 'interface',
                'long', 'native', 'new', 'package', 'private', 'protected', 'public',
                'return', 'short', 'static', 'strictfp', 'super', 'switch',
                'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void',
                'volatile', 'while', 'true', 'false', 'null', '+=', '&&', '-=', '!=',
                '++', '||', '<=', '>=', '<<=', '>>=', '...', '--', '/=', '>>>=', '<<<=',
                'equals', 'inline', 'override']
    
    
    
    @staticmethod
    def remove_split_punctuation_with_floats(text):
        # Replace periods that are not part of a float number with ' . '  <-- Added
        text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)  # <-- Added
        
        # Remove some useless char (excluding '.')
        remove_char = [',', '"', "'", '~']  # <-- Modified (removed '.')
        text = ''.join(char if char not in remove_char else " " for char in text)
        
        # Split text for special characters
        special_char = ['*', '(', ')', '[', ']', '{', '}', ';', '>',
                        '<', '-', '/', ':', '=', '+',
                        '"', "'", '!', '?']
        text = ''.join(char if char not in special_char else " " + char + " " for char in text)
        
        return text


    @staticmethod
    def clean_text(text): #
        
        text = Analyzer2.removeEmojis(text)
        
        # Remove non-ASCII characters
        text = text.encode("ascii", "ignore").decode()
        
        text = text.replace("\n", ' ').replace("\t", ' ').replace("</s>", ' ').replace("<s>", ' ').strip()
        text = " ".join(text.split())
        
        # Remove trailing spaces and multiple white spaces
        text = text.strip()
        text = re.sub(' +', ' ', text)
        
        return text
    
    
    
    @staticmethod
    def filter_comments_by_length(comment):
        words = comment.split()
        if len(words) < 3 or len(words) > 150:
            return True
        return False
    
    
    
    @staticmethod
    def is_english(text):

        # first round of classifiers
        try:
            if detect(text) == "en":
                return True

            langid_prediction = identifier.classify(text)
            if langid_prediction[0] == "en" and langid_prediction[1] > 0.8:
                return True

            cld3_prediction = cld3.get_language(text)
            if cld3_prediction.language == "en" and cld3_prediction.is_reliable:
                return True

        except Exception as e:
            print("classifier round: ", e, text)
            return True

        return True
    
    
    
    @staticmethod
    def apply_cleaning_to_column_safely(df, column_name, new_column_name):
        """
        Cleans a column in the DataFrame:
        - Removes empty lines
        - Preserves indentation and diff markers (+, -,  )
        - Collapses multiple spaces between words
        - Removes trailing whitespace per line
    
        Args:
            df (pd.DataFrame): Input DataFrame
            column_name (str): Input column to clean
            new_column_name (str): Output column for cleaned results
    
        Returns:
            pd.DataFrame: with new_column_name added
        """
        def cleaning_model_input(text):
            text = str(text)
            lines = text.splitlines()
            cleaned_lines = []
    
            for line in lines:
                if not line.strip():
                    continue  # skip empty lines
    
                # Split into diff symbol + indent + content
                match = re.match(r'^([ +-]?)(\s*)(.*)$', line)
                if not match:
                    continue  # skip malformed lines
    
                diff_char, indent, content = match.groups()
    
                # Collapse multiple internal spaces (excluding indent)
                content = re.sub(r'\s{2,}', ' ', content.strip())
    
                # Reconstruct line: diff char + indent + cleaned content
                cleaned_lines.append(f"{diff_char}{indent}{content}")
    
            return "\n".join(cleaned_lines).strip()
    
        def clean_or_empty(val):
            cleaned = cleaning_model_input(val)
            return cleaned if cleaned else ''
    
        if new_column_name in df.columns:
            df = df.drop(columns=[new_column_name])
    
        df[new_column_name] = df[column_name].apply(clean_or_empty)
        return df



    
    @staticmethod
    def normalize_start_end_tokens(df, column_name, new_column_name):
        """
        Replaces START and END tokens with <START> and <END>, preserving inline formatting
        and avoiding replacements inside other words (e.g. LOCATION_END).
        """
    
        def process(val):
            val_str = str(val)
    
            # Replace standalone START and END with <START> and <END>
            val_str = re.sub(r'\bSTART\b', '<START> ', val_str)
            val_str = re.sub(r'\bEND\b', ' <END>', val_str)
    
            has_start = '<START> ' in val_str
            has_end = ' <END>' in val_str
    
            # If still not wrapped, wrap around the entire string
            if not has_start:
                val_str = '<START> ' + val_str
            if not has_end:
                val_str = val_str + ' <END>'
    
            return val_str.strip()
    
        if new_column_name in df.columns:
            df = df.drop(columns=[new_column_name])
    
        df[new_column_name] = df[column_name].apply(process)
        return df


    
    @staticmethod   
    def flatten_to_single_line(df, column_name, new_column_name):
        """
        Converts multiline text to a single line:
        - Removes all newlines
        - Removes indentation
        - Joins lines with space
        - Drops new_column_name if it already exists before creating it
    
        Args:
            df (pd.DataFrame): Input DataFrame
            column_name (str): Input column to flatten
            new_column_name (str): Output column
    
        Returns:
            pd.DataFrame: DataFrame with flattened text column
        """
        def to_single_line(val):
            val_str = str(val)
            lines = val_str.splitlines()
            stripped_lines = [line.strip() for line in lines]
            return ' '.join(stripped_lines).strip()
    
        if new_column_name in df.columns:
            df = df.drop(columns=[new_column_name])
    
        df[new_column_name] = df[column_name].apply(to_single_line)
        return df


    @staticmethod
    def removeEmojis(s):
        # list of all the emoji in unicode
        # source: https://gist.github.com/akkez/99ceeae2f13c9d8d9be7df0279e2c438
        emoji_pattern = regex.compile(
            r"\U0001f469\u200d\u2764\ufe0f\u200d\U0001f48b\u200d\U0001f468|\U0001f468\u200d\u2764\ufe0f\u200d\U0001f48b"
            r"\u200d\U0001f468|\U0001f469\u200d\u2764\ufe0f\u200d\U0001f48b\u200d\U0001f469|\U0001f9d1\U0001f3fb\u200d"
            r"\U0001f91d\u200d\U0001f9d1\U0001f3fb|\U0001f9d1\U0001f3fc\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fb|"
            r"\U0001f9d1\U0001f3fc\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fc|\U0001f9d1\U0001f3fd\u200d\U0001f91d\u200d"
            r"\U0001f9d1\U0001f3fb|\U0001f9d1\U0001f3fd\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fc|\U0001f9d1\U0001f3fd"
            r"\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fd|\U0001f9d1\U0001f3fe\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fb"
            r"|\U0001f9d1\U0001f3fe\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fc|\U0001f9d1\U0001f3fe\u200d\U0001f91d"
            r"\u200d\U0001f9d1\U0001f3fd|\U0001f9d1\U0001f3fe\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fe|\U0001f9d1"
            r"\U0001f3ff\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fb|\U0001f9d1\U0001f3ff\u200d\U0001f91d\u200d\U0001f9d1"
            r"\U0001f3fc|\U0001f9d1\U0001f3ff\u200d\U0001f91d\u200d\U0001f9d1\U0001f3fd|\U0001f9d1\U0001f3ff\u200d"
            r"\U0001f91d\u200d\U0001f9d1\U0001f3fe|\U0001f9d1\U0001f3ff\u200d\U0001f91d\u200d\U0001f9d1\U0001f3ff|"
            r"\U0001f469\U0001f3fc\u200d\U0001f91d\u200d\U0001f469\U0001f3fb|\U0001f469\U0001f3fd\u200d\U0001f91d\u200d"
            r"\U0001f469\U0001f3fb|\U0001f469\U0001f3fd\u200d\U0001f91d\u200d\U0001f469\U0001f3fc|\U0001f469\U0001f3fe"
            r"\u200d\U0001f91d\u200d\U0001f469\U0001f3fb|\U0001f469\U0001f3fe\u200d\U0001f91d\u200d\U0001f469"
            r"\U0001f3fc|\U0001f469\U0001f3fe\u200d\U0001f91d\u200d\U0001f469\U0001f3fd|\U0001f469\U0001f3ff\u200d"
            r"\U0001f91d\u200d\U0001f469\U0001f3fb|\U0001f469\U0001f3ff\u200d\U0001f91d\u200d\U0001f469\U0001f3fc|"
            r"\U0001f469\U0001f3ff\u200d\U0001f91d\u200d\U0001f469\U0001f3fd|\U0001f469\U0001f3ff\u200d\U0001f91d\u200d"
            r"\U0001f469\U0001f3fe|\U0001f469\U0001f3fb\u200d\U0001f91d\u200d\U0001f468\U0001f3fc|\U0001f469\U0001f3fb"
            r"\u200d\U0001f91d\u200d\U0001f468\U0001f3fd|\U0001f469\U0001f3fb\u200d\U0001f91d\u200d\U0001f468"
            r"\U0001f3fe|\U0001f469\U0001f3fb\u200d\U0001f91d\u200d\U0001f468\U0001f3ff|\U0001f469\U0001f3fc\u200d"
            r"\U0001f91d\u200d\U0001f468\U0001f3fb|\U0001f469\U0001f3fc\u200d\U0001f91d\u200d\U0001f468\U0001f3fd|"
            r"\U0001f469\U0001f3fc\u200d\U0001f91d\u200d\U0001f468\U0001f3fe|\U0001f469\U0001f3fc\u200d\U0001f91d\u200d"
            r"\U0001f468\U0001f3ff|\U0001f469\U0001f3fd\u200d\U0001f91d\u200d\U0001f468\U0001f3fb|\U0001f469\U0001f3fd"
            r"\u200d\U0001f91d\u200d\U0001f468\U0001f3fc|\U0001f469\U0001f3fd\u200d\U0001f91d\u200d\U0001f468"
            r"\U0001f3fe|\U0001f469\U0001f3fd\u200d\U0001f91d\u200d\U0001f468\U0001f3ff|\U0001f469\U0001f3fe\u200d"
            r"\U0001f91d\u200d\U0001f468\U0001f3fb|\U0001f469\U0001f3fe\u200d\U0001f91d\u200d\U0001f468\U0001f3fc|"
            r"\U0001f469\U0001f3fe\u200d\U0001f91d\u200d\U0001f468\U0001f3fd|\U0001f469\U0001f3fe\u200d\U0001f91d\u200d"
            r"\U0001f468\U0001f3ff|\U0001f469\U0001f3ff\u200d\U0001f91d\u200d\U0001f468\U0001f3fb|\U0001f469\U0001f3ff"
            r"\u200d\U0001f91d\u200d\U0001f468\U0001f3fc|\U0001f469\U0001f3ff\u200d\U0001f91d\u200d\U0001f468"
            r"\U0001f3fd|\U0001f469\U0001f3ff\u200d\U0001f91d\u200d\U0001f468\U0001f3fe|\U0001f468\U0001f3fc\u200d"
            r"\U0001f91d\u200d\U0001f468\U0001f3fb|\U0001f468\U0001f3fd\u200d\U0001f91d\u200d\U0001f468\U0001f3fb|"
            r"\U0001f468\U0001f3fd\u200d\U0001f91d\u200d\U0001f468\U0001f3fc|\U0001f468\U0001f3fe\u200d\U0001f91d\u200d"
            r"\U0001f468\U0001f3fb|\U0001f468\U0001f3fe\u200d\U0001f91d\u200d\U0001f468\U0001f3fc|\U0001f468\U0001f3fe"
            r"\u200d\U0001f91d\u200d\U0001f468\U0001f3fd|\U0001f468\U0001f3ff\u200d\U0001f91d\u200d\U0001f468"
            r"\U0001f3fb|\U0001f468\U0001f3ff\u200d\U0001f91d\u200d\U0001f468\U0001f3fc|\U0001f468\U0001f3ff\u200d"
            r"\U0001f91d\u200d\U0001f468\U0001f3fd|\U0001f468\U0001f3ff\u200d\U0001f91d\u200d\U0001f468\U0001f3fe|"
            r"\U0001f469\u200d\u2764\u200d\U0001f48b\u200d\U0001f468|\U0001f468\u200d\u2764\u200d\U0001f48b\u200d"
            r"\U0001f468|\U0001f469\u200d\u2764\u200d\U0001f48b\u200d\U0001f469|\U0001f468\u200d\U0001f469\u200d"
            r"\U0001f467\u200d\U0001f466|\U0001f468\u200d\U0001f469\u200d\U0001f466\u200d\U0001f466|\U0001f468\u200d"
            r"\U0001f469\u200d\U0001f467\u200d\U0001f467|\U0001f468\u200d\U0001f468\u200d\U0001f467\u200d\U0001f466|"
            r"\U0001f468\u200d\U0001f468\u200d\U0001f466\u200d\U0001f466|\U0001f468\u200d\U0001f468\u200d\U0001f467"
            r"\u200d\U0001f467|\U0001f469\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466|\U0001f469\u200d\U0001f469"
            r"\u200d\U0001f466\u200d\U0001f466|\U0001f469\u200d\U0001f469\u200d\U0001f467\u200d\U0001f467|\U0001f3f4"
            r"\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f|\U0001f3f4\U000e0067\U000e0062\U000e0073"
            r"\U000e0063\U000e0074\U000e007f|\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f|"
            r"\U0001f469\u200d\u2764\ufe0f\u200d\U0001f468|\U0001f468\u200d\u2764\ufe0f\u200d\U0001f468|\U0001f469"
            r"\u200d\u2764\ufe0f\u200d\U0001f469|\U0001f441\ufe0f\u200d\U0001f5e8\ufe0f|\U0001f471\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f471\U0001f3fc\u200d\u2642\ufe0f|\U0001f471\U0001f3fd\u200d\u2642\ufe0f|\U0001f471"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f471\U0001f3ff\u200d\u2642\ufe0f|\U0001f471\U0001f3fb\u200d\u2640"
            r"\ufe0f|\U0001f471\U0001f3fc\u200d\u2640\ufe0f|\U0001f471\U0001f3fd\u200d\u2640\ufe0f|\U0001f471"
            r"\U0001f3fe\u200d\u2640\ufe0f|\U0001f471\U0001f3ff\u200d\u2640\ufe0f|\U0001f64d\U0001f3fb\u200d\u2642"
            r"\ufe0f|\U0001f64d\U0001f3fc\u200d\u2642\ufe0f|\U0001f64d\U0001f3fd\u200d\u2642\ufe0f|\U0001f64d\U0001f3fe"
            r"\u200d\u2642\ufe0f|\U0001f64d\U0001f3ff\u200d\u2642\ufe0f|\U0001f64d\U0001f3fb\u200d\u2640\ufe0f|"
            r"\U0001f64d\U0001f3fc\u200d\u2640\ufe0f|\U0001f64d\U0001f3fd\u200d\u2640\ufe0f|\U0001f64d\U0001f3fe\u200d"
            r"\u2640\ufe0f|\U0001f64d\U0001f3ff\u200d\u2640\ufe0f|\U0001f64e\U0001f3fb\u200d\u2642\ufe0f|\U0001f64e"
            r"\U0001f3fc\u200d\u2642\ufe0f|\U0001f64e\U0001f3fd\u200d\u2642\ufe0f|\U0001f64e\U0001f3fe\u200d\u2642"
            r"\ufe0f|\U0001f64e\U0001f3ff\u200d\u2642\ufe0f|\U0001f64e\U0001f3fb\u200d\u2640\ufe0f|\U0001f64e\U0001f3fc"
            r"\u200d\u2640\ufe0f|\U0001f64e\U0001f3fd\u200d\u2640\ufe0f|\U0001f64e\U0001f3fe\u200d\u2640\ufe0f|"
            r"\U0001f64e\U0001f3ff\u200d\u2640\ufe0f|\U0001f645\U0001f3fb\u200d\u2642\ufe0f|\U0001f645\U0001f3fc\u200d"
            r"\u2642\ufe0f|\U0001f645\U0001f3fd\u200d\u2642\ufe0f|\U0001f645\U0001f3fe\u200d\u2642\ufe0f|\U0001f645"
            r"\U0001f3ff\u200d\u2642\ufe0f|\U0001f645\U0001f3fb\u200d\u2640\ufe0f|\U0001f645\U0001f3fc\u200d\u2640"
            r"\ufe0f|\U0001f645\U0001f3fd\u200d\u2640\ufe0f|\U0001f645\U0001f3fe\u200d\u2640\ufe0f|\U0001f645\U0001f3ff"
            r"\u200d\u2640\ufe0f|\U0001f646\U0001f3fb\u200d\u2642\ufe0f|\U0001f646\U0001f3fc\u200d\u2642\ufe0f|"
            r"\U0001f646\U0001f3fd\u200d\u2642\ufe0f|\U0001f646\U0001f3fe\u200d\u2642\ufe0f|\U0001f646\U0001f3ff\u200d"
            r"\u2642\ufe0f|\U0001f646\U0001f3fb\u200d\u2640\ufe0f|\U0001f646\U0001f3fc\u200d\u2640\ufe0f|\U0001f646"
            r"\U0001f3fd\u200d\u2640\ufe0f|\U0001f646\U0001f3fe\u200d\u2640\ufe0f|\U0001f646\U0001f3ff\u200d\u2640"
            r"\ufe0f|\U0001f481\U0001f3fb\u200d\u2642\ufe0f|\U0001f481\U0001f3fc\u200d\u2642\ufe0f|\U0001f481\U0001f3fd"
            r"\u200d\u2642\ufe0f|\U0001f481\U0001f3fe\u200d\u2642\ufe0f|\U0001f481\U0001f3ff\u200d\u2642\ufe0f|"
            r"\U0001f481\U0001f3fb\u200d\u2640\ufe0f|\U0001f481\U0001f3fc\u200d\u2640\ufe0f|\U0001f481\U0001f3fd\u200d"
            r"\u2640\ufe0f|\U0001f481\U0001f3fe\u200d\u2640\ufe0f|\U0001f481\U0001f3ff\u200d\u2640\ufe0f|\U0001f64b"
            r"\U0001f3fb\u200d\u2642\ufe0f|\U0001f64b\U0001f3fc\u200d\u2642\ufe0f|\U0001f64b\U0001f3fd\u200d\u2642"
            r"\ufe0f|\U0001f64b\U0001f3fe\u200d\u2642\ufe0f|\U0001f64b\U0001f3ff\u200d\u2642\ufe0f|\U0001f64b\U0001f3fb"
            r"\u200d\u2640\ufe0f|\U0001f64b\U0001f3fc\u200d\u2640\ufe0f|\U0001f64b\U0001f3fd\u200d\u2640\ufe0f|"
            r"\U0001f64b\U0001f3fe\u200d\u2640\ufe0f|\U0001f64b\U0001f3ff\u200d\u2640\ufe0f|\U0001f9cf\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f9cf\U0001f3fc\u200d\u2642\ufe0f|\U0001f9cf\U0001f3fd\u200d\u2642\ufe0f|\U0001f9cf"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f9cf\U0001f3ff\u200d\u2642\ufe0f|\U0001f9cf\U0001f3fb\u200d\u2640"
            r"\ufe0f|\U0001f9cf\U0001f3fc\u200d\u2640\ufe0f|\U0001f9cf\U0001f3fd\u200d\u2640\ufe0f|\U0001f9cf\U0001f3fe"
            r"\u200d\u2640\ufe0f|\U0001f9cf\U0001f3ff\u200d\u2640\ufe0f|\U0001f647\U0001f3fb\u200d\u2642\ufe0f|"
            r"\U0001f647\U0001f3fc\u200d\u2642\ufe0f|\U0001f647\U0001f3fd\u200d\u2642\ufe0f|\U0001f647\U0001f3fe"
            r"\u200d\u2642\ufe0f|\U0001f647\U0001f3ff\u200d\u2642\ufe0f|\U0001f647\U0001f3fb\u200d\u2640\ufe0f|"
            r"\U0001f647\U0001f3fc\u200d\u2640\ufe0f|\U0001f647\U0001f3fd\u200d\u2640\ufe0f|\U0001f647\U0001f3fe\u200d"
            r"\u2640\ufe0f|\U0001f647\U0001f3ff\u200d\u2640\ufe0f|\U0001f926\U0001f3fb\u200d\u2642\ufe0f|\U0001f926"
            r"\U0001f3fc\u200d\u2642\ufe0f|\U0001f926\U0001f3fd\u200d\u2642\ufe0f|\U0001f926\U0001f3fe\u200d\u2642"
            r"\ufe0f|\U0001f926\U0001f3ff\u200d\u2642\ufe0f|\U0001f926\U0001f3fb\u200d\u2640\ufe0f|\U0001f926\U0001f3fc"
            r"\u200d\u2640\ufe0f|\U0001f926\U0001f3fd\u200d\u2640\ufe0f|\U0001f926\U0001f3fe\u200d\u2640\ufe0f|"
            r"\U0001f926\U0001f3ff\u200d\u2640\ufe0f|\U0001f937\U0001f3fb\u200d\u2642\ufe0f|\U0001f937\U0001f3fc\u200d"
            r"\u2642\ufe0f|\U0001f937\U0001f3fd\u200d\u2642\ufe0f|\U0001f937\U0001f3fe\u200d\u2642\ufe0f|\U0001f937"
            r"\U0001f3ff\u200d\u2642\ufe0f|\U0001f937\U0001f3fb\u200d\u2640\ufe0f|\U0001f937\U0001f3fc\u200d\u2640"
            r"\ufe0f|\U0001f937\U0001f3fd\u200d\u2640\ufe0f|\U0001f937\U0001f3fe\u200d\u2640\ufe0f|\U0001f937\U0001f3ff"
            r"\u200d\u2640\ufe0f|\U0001f468\U0001f3fb\u200d\u2695\ufe0f|\U0001f468\U0001f3fc\u200d\u2695\ufe0f|"
            r"\U0001f468\U0001f3fd\u200d\u2695\ufe0f|\U0001f468\U0001f3fe\u200d\u2695\ufe0f|\U0001f468\U0001f3ff\u200d"
            r"\u2695\ufe0f|\U0001f469\U0001f3fb\u200d\u2695\ufe0f|\U0001f469\U0001f3fc\u200d\u2695\ufe0f|\U0001f469"
            r"\U0001f3fd\u200d\u2695\ufe0f|\U0001f469\U0001f3fe\u200d\u2695\ufe0f|\U0001f469\U0001f3ff\u200d\u2695"
            r"\ufe0f|\U0001f468\U0001f3fb\u200d\u2696\ufe0f|\U0001f468\U0001f3fc\u200d\u2696\ufe0f|\U0001f468\U0001f3fd"
            r"\u200d\u2696\ufe0f|\U0001f468\U0001f3fe\u200d\u2696\ufe0f|\U0001f468\U0001f3ff\u200d\u2696\ufe0f|"
            r"\U0001f469\U0001f3fb\u200d\u2696\ufe0f|\U0001f469\U0001f3fc\u200d\u2696\ufe0f|\U0001f469\U0001f3fd\u200d"
            r"\u2696\ufe0f|\U0001f469\U0001f3fe\u200d\u2696\ufe0f|\U0001f469\U0001f3ff\u200d\u2696\ufe0f|\U0001f468"
            r"\U0001f3fb\u200d\u2708\ufe0f|\U0001f468\U0001f3fc\u200d\u2708\ufe0f|\U0001f468\U0001f3fd\u200d\u2708"
            r"\ufe0f|\U0001f468\U0001f3fe\u200d\u2708\ufe0f|\U0001f468\U0001f3ff\u200d\u2708\ufe0f|\U0001f469\U0001f3fb"
            r"\u200d\u2708\ufe0f|\U0001f469\U0001f3fc\u200d\u2708\ufe0f|\U0001f469\U0001f3fd\u200d\u2708\ufe0f|"
            r"\U0001f469\U0001f3fe\u200d\u2708\ufe0f|\U0001f469\U0001f3ff\u200d\u2708\ufe0f|\U0001f46e\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f46e\U0001f3fc\u200d\u2642\ufe0f|\U0001f46e\U0001f3fd\u200d\u2642\ufe0f|\U0001f46e"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f46e\U0001f3ff\u200d\u2642\ufe0f|\U0001f46e\U0001f3fb\u200d\u2640"
            r"\ufe0f|\U0001f46e\U0001f3fc\u200d\u2640\ufe0f|\U0001f46e\U0001f3fd\u200d\u2640\ufe0f|\U0001f46e\U0001f3fe"
            r"\u200d\u2640\ufe0f|\U0001f46e\U0001f3ff\u200d\u2640\ufe0f|\U0001f575\ufe0f\u200d\u2642\ufe0f|\U0001f575"
            r"\U0001f3fb\u200d\u2642\ufe0f|\U0001f575\U0001f3fc\u200d\u2642\ufe0f|\U0001f575\U0001f3fd\u200d\u2642"
            r"\ufe0f|\U0001f575\U0001f3fe\u200d\u2642\ufe0f|\U0001f575\U0001f3ff\u200d\u2642\ufe0f|\U0001f575\ufe0f"
            r"\u200d\u2640\ufe0f|\U0001f575\U0001f3fb\u200d\u2640\ufe0f|\U0001f575\U0001f3fc\u200d\u2640\ufe0f|"
            r"\U0001f575\U0001f3fd\u200d\u2640\ufe0f|\U0001f575\U0001f3fe\u200d\u2640\ufe0f|\U0001f575\U0001f3ff\u200d"
            r"\u2640\ufe0f|\U0001f482\U0001f3fb\u200d\u2642\ufe0f|\U0001f482\U0001f3fc\u200d\u2642\ufe0f|\U0001f482"
            r"\U0001f3fd\u200d\u2642\ufe0f|\U0001f482\U0001f3fe\u200d\u2642\ufe0f|\U0001f482\U0001f3ff\u200d\u2642"
            r"\ufe0f|\U0001f482\U0001f3fb\u200d\u2640\ufe0f|\U0001f482\U0001f3fc\u200d\u2640\ufe0f|\U0001f482\U0001f3fd"
            r"\u200d\u2640\ufe0f|\U0001f482\U0001f3fe\u200d\u2640\ufe0f|\U0001f482\U0001f3ff\u200d\u2640\ufe0f|"
            r"\U0001f477\U0001f3fb\u200d\u2642\ufe0f|\U0001f477\U0001f3fc\u200d\u2642\ufe0f|\U0001f477\U0001f3fd\u200d"
            r"\u2642\ufe0f|\U0001f477\U0001f3fe\u200d\u2642\ufe0f|\U0001f477\U0001f3ff\u200d\u2642\ufe0f|\U0001f477"
            r"\U0001f3fb\u200d\u2640\ufe0f|\U0001f477\U0001f3fc\u200d\u2640\ufe0f|\U0001f477\U0001f3fd\u200d\u2640"
            r"\ufe0f|\U0001f477\U0001f3fe\u200d\u2640\ufe0f|\U0001f477\U0001f3ff\u200d\u2640\ufe0f|\U0001f473\U0001f3fb"
            r"\u200d\u2642\ufe0f|\U0001f473\U0001f3fc\u200d\u2642\ufe0f|\U0001f473\U0001f3fd\u200d\u2642\ufe0f|"
            r"\U0001f473\U0001f3fe\u200d\u2642\ufe0f|\U0001f473\U0001f3ff\u200d\u2642\ufe0f|\U0001f473\U0001f3fb\u200d"
            r"\u2640\ufe0f|\U0001f473\U0001f3fc\u200d\u2640\ufe0f|\U0001f473\U0001f3fd\u200d\u2640\ufe0f|\U0001f473"
            r"\U0001f3fe\u200d\u2640\ufe0f|\U0001f473\U0001f3ff\u200d\u2640\ufe0f|\U0001f9b8\U0001f3fb\u200d\u2642"
            r"\ufe0f|\U0001f9b8\U0001f3fc\u200d\u2642\ufe0f|\U0001f9b8\U0001f3fd\u200d\u2642\ufe0f|\U0001f9b8\U0001f3fe"
            r"\u200d\u2642\ufe0f|\U0001f9b8\U0001f3ff\u200d\u2642\ufe0f|\U0001f9b8\U0001f3fb\u200d\u2640\ufe0f|"
            r"\U0001f9b8\U0001f3fc\u200d\u2640\ufe0f|\U0001f9b8\U0001f3fd\u200d\u2640\ufe0f|\U0001f9b8\U0001f3fe\u200d"
            r"\u2640\ufe0f|\U0001f9b8\U0001f3ff\u200d\u2640\ufe0f|\U0001f9b9\U0001f3fb\u200d\u2642\ufe0f|\U0001f9b9"
            r"\U0001f3fc\u200d\u2642\ufe0f|\U0001f9b9\U0001f3fd\u200d\u2642\ufe0f|\U0001f9b9\U0001f3fe\u200d\u2642"
            r"\ufe0f|\U0001f9b9\U0001f3ff\u200d\u2642\ufe0f|\U0001f9b9\U0001f3fb\u200d\u2640\ufe0f|\U0001f9b9\U0001f3fc"
            r"\u200d\u2640\ufe0f|\U0001f9b9\U0001f3fd\u200d\u2640\ufe0f|\U0001f9b9\U0001f3fe\u200d\u2640\ufe0f|"
            r"\U0001f9b9\U0001f3ff\u200d\u2640\ufe0f|\U0001f9d9\U0001f3fb\u200d\u2642\ufe0f|\U0001f9d9\U0001f3fc\u200d"
            r"\u2642\ufe0f|\U0001f9d9\U0001f3fd\u200d\u2642\ufe0f|\U0001f9d9\U0001f3fe\u200d\u2642\ufe0f|\U0001f9d9"
            r"\U0001f3ff\u200d\u2642\ufe0f|\U0001f9d9\U0001f3fb\u200d\u2640\ufe0f|\U0001f9d9\U0001f3fc\u200d\u2640"
            r"\ufe0f|\U0001f9d9\U0001f3fd\u200d\u2640\ufe0f|\U0001f9d9\U0001f3fe\u200d\u2640\ufe0f|\U0001f9d9\U0001f3ff"
            r"\u200d\u2640\ufe0f|\U0001f9da\U0001f3fb\u200d\u2642\ufe0f|\U0001f9da\U0001f3fc\u200d\u2642\ufe0f|"
            r"\U0001f9da\U0001f3fd\u200d\u2642\ufe0f|\U0001f9da\U0001f3fe\u200d\u2642\ufe0f|\U0001f9da\U0001f3ff\u200d"
            r"\u2642\ufe0f|\U0001f9da\U0001f3fb\u200d\u2640\ufe0f|\U0001f9da\U0001f3fc\u200d\u2640\ufe0f|\U0001f9da"
            r"\U0001f3fd\u200d\u2640\ufe0f|\U0001f9da\U0001f3fe\u200d\u2640\ufe0f|\U0001f9da\U0001f3ff\u200d\u2640"
            r"\ufe0f|\U0001f9db\U0001f3fb\u200d\u2642\ufe0f|\U0001f9db\U0001f3fc\u200d\u2642\ufe0f|\U0001f9db\U0001f3fd"
            r"\u200d\u2642\ufe0f|\U0001f9db\U0001f3fe\u200d\u2642\ufe0f|\U0001f9db\U0001f3ff\u200d\u2642\ufe0f|"
            r"\U0001f9db\U0001f3fb\u200d\u2640\ufe0f|\U0001f9db\U0001f3fc\u200d\u2640\ufe0f|\U0001f9db\U0001f3fd\u200d"
            r"\u2640\ufe0f|\U0001f9db\U0001f3fe\u200d\u2640\ufe0f|\U0001f9db\U0001f3ff\u200d\u2640\ufe0f|\U0001f9dc"
            r"\U0001f3fb\u200d\u2642\ufe0f|\U0001f9dc\U0001f3fc\u200d\u2642\ufe0f|\U0001f9dc\U0001f3fd\u200d\u2642"
            r"\ufe0f|\U0001f9dc\U0001f3fe\u200d\u2642\ufe0f|\U0001f9dc\U0001f3ff\u200d\u2642\ufe0f|\U0001f9dc\U0001f3fb"
            r"\u200d\u2640\ufe0f|\U0001f9dc\U0001f3fc\u200d\u2640\ufe0f|\U0001f9dc\U0001f3fd\u200d\u2640\ufe0f|"
            r"\U0001f9dc\U0001f3fe\u200d\u2640\ufe0f|\U0001f9dc\U0001f3ff\u200d\u2640\ufe0f|\U0001f9dd\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f9dd\U0001f3fc\u200d\u2642\ufe0f|\U0001f9dd\U0001f3fd\u200d\u2642\ufe0f|\U0001f9dd"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f9dd\U0001f3ff\u200d\u2642\ufe0f|\U0001f9dd\U0001f3fb\u200d\u2640"
            r"\ufe0f|\U0001f9dd\U0001f3fc\u200d\u2640\ufe0f|\U0001f9dd\U0001f3fd\u200d\u2640\ufe0f|\U0001f9dd\U0001f3fe"
            r"\u200d\u2640\ufe0f|\U0001f9dd\U0001f3ff\u200d\u2640\ufe0f|\U0001f486\U0001f3fb\u200d\u2642\ufe0f|"
            r"\U0001f486\U0001f3fc\u200d\u2642\ufe0f|\U0001f486\U0001f3fd\u200d\u2642\ufe0f|\U0001f486\U0001f3fe\u200d"
            r"\u2642\ufe0f|\U0001f486\U0001f3ff\u200d\u2642\ufe0f|\U0001f486\U0001f3fb\u200d\u2640\ufe0f|\U0001f486"
            r"\U0001f3fc\u200d\u2640\ufe0f|\U0001f486\U0001f3fd\u200d\u2640\ufe0f|\U0001f486\U0001f3fe\u200d\u2640"
            r"\ufe0f|\U0001f486\U0001f3ff\u200d\u2640\ufe0f|\U0001f487\U0001f3fb\u200d\u2642\ufe0f|\U0001f487\U0001f3fc"
            r"\u200d\u2642\ufe0f|\U0001f487\U0001f3fd\u200d\u2642\ufe0f|\U0001f487\U0001f3fe\u200d\u2642\ufe0f|"
            r"\U0001f487\U0001f3ff\u200d\u2642\ufe0f|\U0001f487\U0001f3fb\u200d\u2640\ufe0f|\U0001f487\U0001f3fc\u200d"
            r"\u2640\ufe0f|\U0001f487\U0001f3fd\u200d\u2640\ufe0f|\U0001f487\U0001f3fe\u200d\u2640\ufe0f|\U0001f487"
            r"\U0001f3ff\u200d\u2640\ufe0f|\U0001f6b6\U0001f3fb\u200d\u2642\ufe0f|\U0001f6b6\U0001f3fc\u200d\u2642"
            r"\ufe0f|\U0001f6b6\U0001f3fd\u200d\u2642\ufe0f|\U0001f6b6\U0001f3fe\u200d\u2642\ufe0f|\U0001f6b6\U0001f3ff"
            r"\u200d\u2642\ufe0f|\U0001f6b6\U0001f3fb\u200d\u2640\ufe0f|\U0001f6b6\U0001f3fc\u200d\u2640\ufe0f|"
            r"\U0001f6b6\U0001f3fd\u200d\u2640\ufe0f|\U0001f6b6\U0001f3fe\u200d\u2640\ufe0f|\U0001f6b6\U0001f3ff\u200d"
            r"\u2640\ufe0f|\U0001f9cd\U0001f3fb\u200d\u2642\ufe0f|\U0001f9cd\U0001f3fc\u200d\u2642\ufe0f|\U0001f9cd"
            r"\U0001f3fd\u200d\u2642\ufe0f|\U0001f9cd\U0001f3fe\u200d\u2642\ufe0f|\U0001f9cd\U0001f3ff\u200d\u2642"
            r"\ufe0f|\U0001f9cd\U0001f3fb\u200d\u2640\ufe0f|\U0001f9cd\U0001f3fc\u200d\u2640\ufe0f|\U0001f9cd\U0001f3fd"
            r"\u200d\u2640\ufe0f|\U0001f9cd\U0001f3fe\u200d\u2640\ufe0f|\U0001f9cd\U0001f3ff\u200d\u2640\ufe0f|"
            r"\U0001f9ce\U0001f3fb\u200d\u2642\ufe0f|\U0001f9ce\U0001f3fc\u200d\u2642\ufe0f|\U0001f9ce\U0001f3fd\u200d"
            r"\u2642\ufe0f|\U0001f9ce\U0001f3fe\u200d\u2642\ufe0f|\U0001f9ce\U0001f3ff\u200d\u2642\ufe0f|\U0001f9ce"
            r"\U0001f3fb\u200d\u2640\ufe0f|\U0001f9ce\U0001f3fc\u200d\u2640\ufe0f|\U0001f9ce\U0001f3fd\u200d\u2640"
            r"\ufe0f|\U0001f9ce\U0001f3fe\u200d\u2640\ufe0f|\U0001f9ce\U0001f3ff\u200d\u2640\ufe0f|\U0001f3c3\U0001f3fb"
            r"\u200d\u2642\ufe0f|\U0001f3c3\U0001f3fc\u200d\u2642\ufe0f|\U0001f3c3\U0001f3fd\u200d\u2642\ufe0f|"
            r"\U0001f3c3\U0001f3fe\u200d\u2642\ufe0f|\U0001f3c3\U0001f3ff\u200d\u2642\ufe0f|\U0001f3c3\U0001f3fb\u200d"
            r"\u2640\ufe0f|\U0001f3c3\U0001f3fc\u200d\u2640\ufe0f|\U0001f3c3\U0001f3fd\u200d\u2640\ufe0f|\U0001f3c3"
            r"\U0001f3fe\u200d\u2640\ufe0f|\U0001f3c3\U0001f3ff\u200d\u2640\ufe0f|\U0001f9d6\U0001f3fb\u200d\u2642"
            r"\ufe0f|\U0001f9d6\U0001f3fc\u200d\u2642\ufe0f|\U0001f9d6\U0001f3fd\u200d\u2642\ufe0f|\U0001f9d6\U0001f3fe"
            r"\u200d\u2642\ufe0f|\U0001f9d6\U0001f3ff\u200d\u2642\ufe0f|\U0001f9d6\U0001f3fb\u200d\u2640\ufe0f|"
            r"\U0001f9d6\U0001f3fc\u200d\u2640\ufe0f|\U0001f9d6\U0001f3fd\u200d\u2640\ufe0f|\U0001f9d6\U0001f3fe\u200d"
            r"\u2640\ufe0f|\U0001f9d6\U0001f3ff\u200d\u2640\ufe0f|\U0001f9d7\U0001f3fb\u200d\u2642\ufe0f|\U0001f9d7"
            r"\U0001f3fc\u200d\u2642\ufe0f|\U0001f9d7\U0001f3fd\u200d\u2642\ufe0f|\U0001f9d7\U0001f3fe\u200d\u2642"
            r"\ufe0f|\U0001f9d7\U0001f3ff\u200d\u2642\ufe0f|\U0001f9d7\U0001f3fb\u200d\u2640\ufe0f|\U0001f9d7\U0001f3fc"
            r"\u200d\u2640\ufe0f|\U0001f9d7\U0001f3fd\u200d\u2640\ufe0f|\U0001f9d7\U0001f3fe\u200d\u2640\ufe0f|"
            r"\U0001f9d7\U0001f3ff\u200d\u2640\ufe0f|\U0001f3cc\ufe0f\u200d\u2642\ufe0f|\U0001f3cc\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f3cc\U0001f3fc\u200d\u2642\ufe0f|\U0001f3cc\U0001f3fd\u200d\u2642\ufe0f|\U0001f3cc"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f3cc\U0001f3ff\u200d\u2642\ufe0f|\U0001f3cc\ufe0f\u200d\u2640\ufe0f|"
            r"\U0001f3cc\U0001f3fb\u200d\u2640\ufe0f|\U0001f3cc\U0001f3fc\u200d\u2640\ufe0f|\U0001f3cc\U0001f3fd\u200d"
            r"\u2640\ufe0f|\U0001f3cc\U0001f3fe\u200d\u2640\ufe0f|\U0001f3cc\U0001f3ff\u200d\u2640\ufe0f|\U0001f3c4"
            r"\U0001f3fb\u200d\u2642\ufe0f|\U0001f3c4\U0001f3fc\u200d\u2642\ufe0f|\U0001f3c4\U0001f3fd\u200d\u2642"
            r"\ufe0f|\U0001f3c4\U0001f3fe\u200d\u2642\ufe0f|\U0001f3c4\U0001f3ff\u200d\u2642\ufe0f|\U0001f3c4\U0001f3fb"
            r"\u200d\u2640\ufe0f|\U0001f3c4\U0001f3fc\u200d\u2640\ufe0f|\U0001f3c4\U0001f3fd\u200d\u2640\ufe0f|"
            r"\U0001f3c4\U0001f3fe\u200d\u2640\ufe0f|\U0001f3c4\U0001f3ff\u200d\u2640\ufe0f|\U0001f6a3\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f6a3\U0001f3fc\u200d\u2642\ufe0f|\U0001f6a3\U0001f3fd\u200d\u2642\ufe0f|\U0001f6a3"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f6a3\U0001f3ff\u200d\u2642\ufe0f|\U0001f6a3\U0001f3fb\u200d\u2640"
            r"\ufe0f|\U0001f6a3\U0001f3fc\u200d\u2640\ufe0f|\U0001f6a3\U0001f3fd\u200d\u2640\ufe0f|\U0001f6a3\U0001f3fe"
            r"\u200d\u2640\ufe0f|\U0001f6a3\U0001f3ff\u200d\u2640\ufe0f|\U0001f3ca\U0001f3fb\u200d\u2642\ufe0f|"
            r"\U0001f3ca\U0001f3fc\u200d\u2642\ufe0f|\U0001f3ca\U0001f3fd\u200d\u2642\ufe0f|\U0001f3ca\U0001f3fe\u200d"
            r"\u2642\ufe0f|\U0001f3ca\U0001f3ff\u200d\u2642\ufe0f|\U0001f3ca\U0001f3fb\u200d\u2640\ufe0f|\U0001f3ca"
            r"\U0001f3fc\u200d\u2640\ufe0f|\U0001f3ca\U0001f3fd\u200d\u2640\ufe0f|\U0001f3ca\U0001f3fe\u200d\u2640"
            r"\ufe0f|\U0001f3ca\U0001f3ff\u200d\u2640\ufe0f|\u26f9\ufe0f\u200d\u2642\ufe0f|\u26f9\U0001f3fb\u200d\u2642"
            r"\ufe0f|\u26f9\U0001f3fc\u200d\u2642\ufe0f|\u26f9\U0001f3fd\u200d\u2642\ufe0f|\u26f9\U0001f3fe\u200d\u2642"
            r"\ufe0f|\u26f9\U0001f3ff\u200d\u2642\ufe0f|\u26f9\ufe0f\u200d\u2640\ufe0f|\u26f9\U0001f3fb\u200d\u2640"
            r"\ufe0f|\u26f9\U0001f3fc\u200d\u2640\ufe0f|\u26f9\U0001f3fd\u200d\u2640\ufe0f|\u26f9\U0001f3fe\u200d\u2640"
            r"\ufe0f|\u26f9\U0001f3ff\u200d\u2640\ufe0f|\U0001f3cb\ufe0f\u200d\u2642\ufe0f|\U0001f3cb\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f3cb\U0001f3fc\u200d\u2642\ufe0f|\U0001f3cb\U0001f3fd\u200d\u2642\ufe0f|\U0001f3cb"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f3cb\U0001f3ff\u200d\u2642\ufe0f|\U0001f3cb\ufe0f\u200d\u2640\ufe0f|"
            r"\U0001f3cb\U0001f3fb\u200d\u2640\ufe0f|\U0001f3cb\U0001f3fc\u200d\u2640\ufe0f|\U0001f3cb\U0001f3fd\u200d"
            r"\u2640\ufe0f|\U0001f3cb\U0001f3fe\u200d\u2640\ufe0f|\U0001f3cb\U0001f3ff\u200d\u2640\ufe0f|\U0001f6b4"
            r"\U0001f3fb\u200d\u2642\ufe0f|\U0001f6b4\U0001f3fc\u200d\u2642\ufe0f|\U0001f6b4\U0001f3fd\u200d\u2642"
            r"\ufe0f|\U0001f6b4\U0001f3fe\u200d\u2642\ufe0f|\U0001f6b4\U0001f3ff\u200d\u2642\ufe0f|\U0001f6b4\U0001f3fb"
            r"\u200d\u2640\ufe0f|\U0001f6b4\U0001f3fc\u200d\u2640\ufe0f|\U0001f6b4\U0001f3fd\u200d\u2640\ufe0f|"
            r"\U0001f6b4\U0001f3fe\u200d\u2640\ufe0f|\U0001f6b4\U0001f3ff\u200d\u2640\ufe0f|\U0001f6b5\U0001f3fb\u200d"
            r"\u2642\ufe0f|\U0001f6b5\U0001f3fc\u200d\u2642\ufe0f|\U0001f6b5\U0001f3fd\u200d\u2642\ufe0f|\U0001f6b5"
            r"\U0001f3fe\u200d\u2642\ufe0f|\U0001f6b5\U0001f3ff\u200d\u2642\ufe0f|\U0001f6b5\U0001f3fb\u200d\u2640"
            r"\ufe0f|\U0001f6b5\U0001f3fc\u200d\u2640\ufe0f|\U0001f6b5\U0001f3fd\u200d\u2640\ufe0f|\U0001f6b5\U0001f3fe"
            r"\u200d\u2640\ufe0f|\U0001f6b5\U0001f3ff\u200d\u2640\ufe0f|\U0001f938\U0001f3fb\u200d\u2642\ufe0f|"
            r"\U0001f938\U0001f3fc\u200d\u2642\ufe0f|\U0001f938\U0001f3fd\u200d\u2642\ufe0f|\U0001f938\U0001f3fe\u200d"
            r"\u2642\ufe0f|\U0001f938\U0001f3ff\u200d\u2642\ufe0f|\U0001f938\U0001f3fb\u200d\u2640\ufe0f|\U0001f938"
            r"\U0001f3fc\u200d\u2640\ufe0f|\U0001f938\U0001f3fd\u200d\u2640\ufe0f|\U0001f938\U0001f3fe\u200d\u2640"
            r"\ufe0f|\U0001f938\U0001f3ff\u200d\u2640\ufe0f|\U0001f93d\U0001f3fb\u200d\u2642\ufe0f|\U0001f93d\U0001f3fc"
            r"\u200d\u2642\ufe0f|\U0001f93d\U0001f3fd\u200d\u2642\ufe0f|\U0001f93d\U0001f3fe\u200d\u2642\ufe0f|"
            r"\U0001f93d\U0001f3ff\u200d\u2642\ufe0f|\U0001f93d\U0001f3fb\u200d\u2640\ufe0f|\U0001f93d\U0001f3fc\u200d"
            r"\u2640\ufe0f|\U0001f93d\U0001f3fd\u200d\u2640\ufe0f|\U0001f93d\U0001f3fe\u200d\u2640\ufe0f|\U0001f93d"
            r"\U0001f3ff\u200d\u2640\ufe0f|\U0001f93e\U0001f3fb\u200d\u2642\ufe0f|\U0001f93e\U0001f3fc\u200d\u2642"
            r"\ufe0f|\U0001f93e\U0001f3fd\u200d\u2642\ufe0f|\U0001f93e\U0001f3fe\u200d\u2642\ufe0f|\U0001f93e\U0001f3ff"
            r"\u200d\u2642\ufe0f|\U0001f93e\U0001f3fb\u200d\u2640\ufe0f|\U0001f93e\U0001f3fc\u200d\u2640\ufe0f|"
            r"\U0001f93e\U0001f3fd\u200d\u2640\ufe0f|\U0001f93e\U0001f3fe\u200d\u2640\ufe0f|\U0001f93e\U0001f3ff\u200d"
            r"\u2640\ufe0f|\U0001f939\U0001f3fb\u200d\u2642\ufe0f|\U0001f939\U0001f3fc\u200d\u2642\ufe0f|\U0001f939"
            r"\U0001f3fd\u200d\u2642\ufe0f|\U0001f939\U0001f3fe\u200d\u2642\ufe0f|\U0001f939\U0001f3ff\u200d\u2642"
            r"\ufe0f|\U0001f939\U0001f3fb\u200d\u2640\ufe0f|\U0001f939\U0001f3fc\u200d\u2640\ufe0f|\U0001f939\U0001f3fd"
            r"\u200d\u2640\ufe0f|\U0001f939\U0001f3fe\u200d\u2640\ufe0f|\U0001f939\U0001f3ff\u200d\u2640\ufe0f|"
            r"\U0001f9d8\U0001f3fb\u200d\u2642\ufe0f|\U0001f9d8\U0001f3fc\u200d\u2642\ufe0f|\U0001f9d8\U0001f3fd\u200d"
            r"\u2642\ufe0f|\U0001f9d8\U0001f3fe\u200d\u2642\ufe0f|\U0001f9d8\U0001f3ff\u200d\u2642\ufe0f|\U0001f9d8"
            r"\U0001f3fb\u200d\u2640\ufe0f|\U0001f9d8\U0001f3fc\u200d\u2640\ufe0f|\U0001f9d8\U0001f3fd\u200d\u2640"
            r"\ufe0f|\U0001f9d8\U0001f3fe\u200d\u2640\ufe0f|\U0001f9d8\U0001f3ff\u200d\u2640\ufe0f|\U0001f9d1\u200d"
            r"\U0001f91d\u200d\U0001f9d1|\U0001f469\u200d\u2764\u200d\U0001f468|\U0001f468\u200d\u2764\u200d\U0001f468|"
            r"\U0001f469\u200d\u2764\u200d\U0001f469|\U0001f468\u200d\U0001f469\u200d\U0001f466|\U0001f468\u200d"
            r"\U0001f469\u200d\U0001f467|\U0001f468\u200d\U0001f468\u200d\U0001f466|\U0001f468\u200d\U0001f468\u200d"
            r"\U0001f467|\U0001f469\u200d\U0001f469\u200d\U0001f466|\U0001f469\u200d\U0001f469\u200d\U0001f467|"
            r"\U0001f468\u200d\U0001f466\u200d\U0001f466|\U0001f468\u200d\U0001f467\u200d\U0001f466|\U0001f468\u200d"
            r"\U0001f467\u200d\U0001f467|\U0001f469\u200d\U0001f466\u200d\U0001f466|\U0001f469\u200d\U0001f467\u200d"
            r"\U0001f466|\U0001f469\u200d\U0001f467\u200d\U0001f467|\U0001f441\u200d\U0001f5e8\ufe0f|\U0001f441\ufe0f"
            r"\u200d\U0001f5e8|\U0001f471\u200d\u2642\ufe0f|\U0001f471\U0001f3fb\u200d\u2642|\U0001f471\U0001f3fc\u200d"
            r"\u2642|\U0001f471\U0001f3fd\u200d\u2642|\U0001f471\U0001f3fe\u200d\u2642|\U0001f471\U0001f3ff\u200d"
            r"\u2642|\U0001f468\U0001f3fb\u200d\U0001f9b0|\U0001f468\U0001f3fc\u200d\U0001f9b0|\U0001f468\U0001f3fd"
            r"\u200d\U0001f9b0|\U0001f468\U0001f3fe\u200d\U0001f9b0|\U0001f468\U0001f3ff\u200d\U0001f9b0|\U0001f468"
            r"\U0001f3fb\u200d\U0001f9b1|\U0001f468\U0001f3fc\u200d\U0001f9b1|\U0001f468\U0001f3fd\u200d\U0001f9b1|"
            r"\U0001f468\U0001f3fe\u200d\U0001f9b1|\U0001f468\U0001f3ff\u200d\U0001f9b1|\U0001f468\U0001f3fb\u200d"
            r"\U0001f9b3|\U0001f468\U0001f3fc\u200d\U0001f9b3|\U0001f468\U0001f3fd\u200d\U0001f9b3|\U0001f468\U0001f3fe"
            r"\u200d\U0001f9b3|\U0001f468\U0001f3ff\u200d\U0001f9b3|\U0001f468\U0001f3fb\u200d\U0001f9b2|\U0001f468"
            r"\U0001f3fc\u200d\U0001f9b2|\U0001f468\U0001f3fd\u200d\U0001f9b2|\U0001f468\U0001f3fe\u200d\U0001f9b2|"
            r"\U0001f468\U0001f3ff\u200d\U0001f9b2|\U0001f471\u200d\u2640\ufe0f|\U0001f471\U0001f3fb\u200d\u2640|"
            r"\U0001f471\U0001f3fc\u200d\u2640|\U0001f471\U0001f3fd\u200d\u2640|\U0001f471\U0001f3fe\u200d\u2640|"
            r"\U0001f471\U0001f3ff\u200d\u2640|\U0001f469\U0001f3fb\u200d\U0001f9b0|\U0001f469\U0001f3fc\u200d"
            r"\U0001f9b0|\U0001f469\U0001f3fd\u200d\U0001f9b0|\U0001f469\U0001f3fe\u200d\U0001f9b0|\U0001f469\U0001f3ff"
            r"\u200d\U0001f9b0|\U0001f469\U0001f3fb\u200d\U0001f9b1|\U0001f469\U0001f3fc\u200d\U0001f9b1|\U0001f469"
            r"\U0001f3fd\u200d\U0001f9b1|\U0001f469\U0001f3fe\u200d\U0001f9b1|\U0001f469\U0001f3ff\u200d\U0001f9b1|"
            r"\U0001f469\U0001f3fb\u200d\U0001f9b3|\U0001f469\U0001f3fc\u200d\U0001f9b3|\U0001f469\U0001f3fd\u200d"
            r"\U0001f9b3|\U0001f469\U0001f3fe\u200d\U0001f9b3|\U0001f469\U0001f3ff\u200d\U0001f9b3|\U0001f469\U0001f3fb"
            r"\u200d\U0001f9b2|\U0001f469\U0001f3fc\u200d\U0001f9b2|\U0001f469\U0001f3fd\u200d\U0001f9b2|\U0001f469"
            r"\U0001f3fe\u200d\U0001f9b2|\U0001f469\U0001f3ff\u200d\U0001f9b2|\U0001f64d\u200d\u2642\ufe0f|\U0001f64d"
            r"\U0001f3fb\u200d\u2642|\U0001f64d\U0001f3fc\u200d\u2642|\U0001f64d\U0001f3fd\u200d\u2642|\U0001f64d"
            r"\U0001f3fe\u200d\u2642|\U0001f64d\U0001f3ff\u200d\u2642|\U0001f64d\u200d\u2640\ufe0f|\U0001f64d\U0001f3fb"
            r"\u200d\u2640|\U0001f64d\U0001f3fc\u200d\u2640|\U0001f64d\U0001f3fd\u200d\u2640|\U0001f64d\U0001f3fe\u200d"
            r"\u2640|\U0001f64d\U0001f3ff\u200d\u2640|\U0001f64e\u200d\u2642\ufe0f|\U0001f64e\U0001f3fb\u200d\u2642|"
            r"\U0001f64e\U0001f3fc\u200d\u2642|\U0001f64e\U0001f3fd\u200d\u2642|\U0001f64e\U0001f3fe\u200d\u2642|"
            r"\U0001f64e\U0001f3ff\u200d\u2642|\U0001f64e\u200d\u2640\ufe0f|\U0001f64e\U0001f3fb\u200d\u2640|\U0001f64e"
            r"\U0001f3fc\u200d\u2640|\U0001f64e\U0001f3fd\u200d\u2640|\U0001f64e\U0001f3fe\u200d\u2640|\U0001f64e"
            r"\U0001f3ff\u200d\u2640|\U0001f645\u200d\u2642\ufe0f|\U0001f645\U0001f3fb\u200d\u2642|\U0001f645\U0001f3fc"
            r"\u200d\u2642|\U0001f645\U0001f3fd\u200d\u2642|\U0001f645\U0001f3fe\u200d\u2642|\U0001f645\U0001f3ff\u200d"
            r"\u2642|\U0001f645\u200d\u2640\ufe0f|\U0001f645\U0001f3fb\u200d\u2640|\U0001f645\U0001f3fc\u200d\u2640|"
            r"\U0001f645\U0001f3fd\u200d\u2640|\U0001f645\U0001f3fe\u200d\u2640|\U0001f645\U0001f3ff\u200d\u2640|"
            r"\U0001f646\u200d\u2642\ufe0f|\U0001f646\U0001f3fb\u200d\u2642|\U0001f646\U0001f3fc\u200d\u2642|\U0001f646"
            r"\U0001f3fd\u200d\u2642|\U0001f646\U0001f3fe\u200d\u2642|\U0001f646\U0001f3ff\u200d\u2642|\U0001f646\u200d"
            r"\u2640\ufe0f|\U0001f646\U0001f3fb\u200d\u2640|\U0001f646\U0001f3fc\u200d\u2640|\U0001f646\U0001f3fd\u200d"
            r"\u2640|\U0001f646\U0001f3fe\u200d\u2640|\U0001f646\U0001f3ff\u200d\u2640|\U0001f481\u200d\u2642\ufe0f|"
            r"\U0001f481\U0001f3fb\u200d\u2642|\U0001f481\U0001f3fc\u200d\u2642|\U0001f481\U0001f3fd\u200d\u2642|"
            r"\U0001f481\U0001f3fe\u200d\u2642|\U0001f481\U0001f3ff\u200d\u2642|\U0001f481\u200d\u2640\ufe0f|\U0001f481"
            r"\U0001f3fb\u200d\u2640|\U0001f481\U0001f3fc\u200d\u2640|\U0001f481\U0001f3fd\u200d\u2640|\U0001f481"
            r"\U0001f3fe\u200d\u2640|\U0001f481\U0001f3ff\u200d\u2640|\U0001f64b\u200d\u2642\ufe0f|\U0001f64b\U0001f3fb"
            r"\u200d\u2642|\U0001f64b\U0001f3fc\u200d\u2642|\U0001f64b\U0001f3fd\u200d\u2642|\U0001f64b\U0001f3fe\u200d"
            r"\u2642|\U0001f64b\U0001f3ff\u200d\u2642|\U0001f64b\u200d\u2640\ufe0f|\U0001f64b\U0001f3fb\u200d\u2640|"
            r"\U0001f64b\U0001f3fc\u200d\u2640|\U0001f64b\U0001f3fd\u200d\u2640|\U0001f64b\U0001f3fe\u200d\u2640|"
            r"\U0001f64b\U0001f3ff\u200d\u2640|\U0001f9cf\u200d\u2642\ufe0f|\U0001f9cf\U0001f3fb\u200d\u2642|\U0001f9cf"
            r"\U0001f3fc\u200d\u2642|\U0001f9cf\U0001f3fd\u200d\u2642|\U0001f9cf\U0001f3fe\u200d\u2642|\U0001f9cf"
            r"\U0001f3ff\u200d\u2642|\U0001f9cf\u200d\u2640\ufe0f|\U0001f9cf\U0001f3fb\u200d\u2640|\U0001f9cf\U0001f3fc"
            r"\u200d\u2640|\U0001f9cf\U0001f3fd\u200d\u2640|\U0001f9cf\U0001f3fe\u200d\u2640|\U0001f9cf\U0001f3ff\u200d"
            r"\u2640|\U0001f647\u200d\u2642\ufe0f|\U0001f647\U0001f3fb\u200d\u2642|\U0001f647\U0001f3fc\u200d\u2642|"
            r"\U0001f647\U0001f3fd\u200d\u2642|\U0001f647\U0001f3fe\u200d\u2642|\U0001f647\U0001f3ff\u200d\u2642|"
            r"\U0001f647\u200d\u2640\ufe0f|\U0001f647\U0001f3fb\u200d\u2640|\U0001f647\U0001f3fc\u200d\u2640|\U0001f647"
            r"\U0001f3fd\u200d\u2640|\U0001f647\U0001f3fe\u200d\u2640|\U0001f647\U0001f3ff\u200d\u2640|\U0001f926\u200d"
            r"\u2642\ufe0f|\U0001f926\U0001f3fb\u200d\u2642|\U0001f926\U0001f3fc\u200d\u2642|\U0001f926\U0001f3fd\u200d"
            r"\u2642|\U0001f926\U0001f3fe\u200d\u2642|\U0001f926\U0001f3ff\u200d\u2642|\U0001f926\u200d\u2640\ufe0f|"
            r"\U0001f926\U0001f3fb\u200d\u2640|\U0001f926\U0001f3fc\u200d\u2640|\U0001f926\U0001f3fd\u200d\u2640|"
            r"\U0001f926\U0001f3fe\u200d\u2640|\U0001f926\U0001f3ff\u200d\u2640|\U0001f937\u200d\u2642\ufe0f|\U0001f937"
            r"\U0001f3fb\u200d\u2642|\U0001f937\U0001f3fc\u200d\u2642|\U0001f937\U0001f3fd\u200d\u2642|\U0001f937"
            r"\U0001f3fe\u200d\u2642|\U0001f937\U0001f3ff\u200d\u2642|\U0001f937\u200d\u2640\ufe0f|\U0001f937\U0001f3fb"
            r"\u200d\u2640|\U0001f937\U0001f3fc\u200d\u2640|\U0001f937\U0001f3fd\u200d\u2640|\U0001f937\U0001f3fe\u200d"
            r"\u2640|\U0001f937\U0001f3ff\u200d\u2640|\U0001f468\u200d\u2695\ufe0f|\U0001f468\U0001f3fb\u200d\u2695|"
            r"\U0001f468\U0001f3fc\u200d\u2695|\U0001f468\U0001f3fd\u200d\u2695|\U0001f468\U0001f3fe\u200d\u2695|"
            r"\U0001f468\U0001f3ff\u200d\u2695|\U0001f469\u200d\u2695\ufe0f|\U0001f469\U0001f3fb\u200d\u2695|\U0001f469"
            r"\U0001f3fc\u200d\u2695|\U0001f469\U0001f3fd\u200d\u2695|\U0001f469\U0001f3fe\u200d\u2695|\U0001f469"
            r"\U0001f3ff\u200d\u2695|\U0001f468\U0001f3fb\u200d\U0001f393|\U0001f468\U0001f3fc\u200d\U0001f393|"
            r"\U0001f468\U0001f3fd\u200d\U0001f393|\U0001f468\U0001f3fe\u200d\U0001f393|\U0001f468\U0001f3ff\u200d"
            r"\U0001f393|\U0001f469\U0001f3fb\u200d\U0001f393|\U0001f469\U0001f3fc\u200d\U0001f393|\U0001f469\U0001f3fd"
            r"\u200d\U0001f393|\U0001f469\U0001f3fe\u200d\U0001f393|\U0001f469\U0001f3ff\u200d\U0001f393|\U0001f468"
            r"\U0001f3fb\u200d\U0001f3eb|\U0001f468\U0001f3fc\u200d\U0001f3eb|\U0001f468\U0001f3fd\u200d\U0001f3eb|"
            r"\U0001f468\U0001f3fe\u200d\U0001f3eb|\U0001f468\U0001f3ff\u200d\U0001f3eb|\U0001f469\U0001f3fb\u200d"
            r"\U0001f3eb|\U0001f469\U0001f3fc\u200d\U0001f3eb|\U0001f469\U0001f3fd\u200d\U0001f3eb|\U0001f469\U0001f3fe"
            r"\u200d\U0001f3eb|\U0001f469\U0001f3ff\u200d\U0001f3eb|\U0001f468\u200d\u2696\ufe0f|\U0001f468\U0001f3fb"
            r"\u200d\u2696|\U0001f468\U0001f3fc\u200d\u2696|\U0001f468\U0001f3fd\u200d\u2696|\U0001f468\U0001f3fe\u200d"
            r"\u2696|\U0001f468\U0001f3ff\u200d\u2696|\U0001f469\u200d\u2696\ufe0f|\U0001f469\U0001f3fb\u200d\u2696|"
            r"\U0001f469\U0001f3fc\u200d\u2696|\U0001f469\U0001f3fd\u200d\u2696|\U0001f469\U0001f3fe\u200d\u2696|"
            r"\U0001f469\U0001f3ff\u200d\u2696|\U0001f468\U0001f3fb\u200d\U0001f33e|\U0001f468\U0001f3fc\u200d\U0001f33"
            r"e|\U0001f468\U0001f3fd\u200d\U0001f33e|\U0001f468\U0001f3fe\u200d\U0001f33e|\U0001f468\U0001f3ff\u200d"
            r"\U0001f33e|\U0001f469\U0001f3fb\u200d\U0001f33e|\U0001f469\U0001f3fc\u200d\U0001f33e|\U0001f469\U0001f3fd"
            r"\u200d\U0001f33e|\U0001f469\U0001f3fe\u200d\U0001f33e|\U0001f469\U0001f3ff\u200d\U0001f33e|\U0001f468"
            r"\U0001f3fb\u200d\U0001f373|\U0001f468\U0001f3fc\u200d\U0001f373|\U0001f468\U0001f3fd\u200d\U0001f373|"
            r"\U0001f468\U0001f3fe\u200d\U0001f373|\U0001f468\U0001f3ff\u200d\U0001f373|\U0001f469\U0001f3fb\u200d"
            r"\U0001f373|\U0001f469\U0001f3fc\u200d\U0001f373|\U0001f469\U0001f3fd\u200d\U0001f373|\U0001f469\U0001f3fe"
            r"\u200d\U0001f373|\U0001f469\U0001f3ff\u200d\U0001f373|\U0001f468\U0001f3fb\u200d\U0001f527|\U0001f468"
            r"\U0001f3fc\u200d\U0001f527|\U0001f468\U0001f3fd\u200d\U0001f527|\U0001f468\U0001f3fe\u200d\U0001f527|"
            r"\U0001f468\U0001f3ff\u200d\U0001f527|\U0001f469\U0001f3fb\u200d\U0001f527|\U0001f469\U0001f3fc\u200d"
            r"\U0001f527|\U0001f469\U0001f3fd\u200d\U0001f527|\U0001f469\U0001f3fe\u200d\U0001f527|\U0001f469\U0001f3ff"
            r"\u200d\U0001f527|\U0001f468\U0001f3fb\u200d\U0001f3ed|\U0001f468\U0001f3fc\u200d\U0001f3ed|\U0001f468"
            r"\U0001f3fd\u200d\U0001f3ed|\U0001f468\U0001f3fe\u200d\U0001f3ed|\U0001f468\U0001f3ff\u200d\U0001f3ed|"
            r"\U0001f469\U0001f3fb\u200d\U0001f3ed|\U0001f469\U0001f3fc\u200d\U0001f3ed|\U0001f469\U0001f3fd\u200d"
            r"\U0001f3ed|\U0001f469\U0001f3fe\u200d\U0001f3ed|\U0001f469\U0001f3ff\u200d\U0001f3ed|\U0001f468\U0001f3fb"
            r"\u200d\U0001f4bc|\U0001f468\U0001f3fc\u200d\U0001f4bc|\U0001f468\U0001f3fd\u200d\U0001f4bc|\U0001f468"
            r"\U0001f3fe\u200d\U0001f4bc|\U0001f468\U0001f3ff\u200d\U0001f4bc|\U0001f469\U0001f3fb\u200d\U0001f4bc|"
            r"\U0001f469\U0001f3fc\u200d\U0001f4bc|\U0001f469\U0001f3fd\u200d\U0001f4bc|\U0001f469\U0001f3fe\u200d"
            r"\U0001f4bc|\U0001f469\U0001f3ff\u200d\U0001f4bc|\U0001f468\U0001f3fb\u200d\U0001f52c|\U0001f468\U0001f3fc"
            r"\u200d\U0001f52c|\U0001f468\U0001f3fd\u200d\U0001f52c|\U0001f468\U0001f3fe\u200d\U0001f52c|\U0001f468"
            r"\U0001f3ff\u200d\U0001f52c|\U0001f469\U0001f3fb\u200d\U0001f52c|\U0001f469\U0001f3fc\u200d\U0001f52c|"
            r"\U0001f469\U0001f3fd\u200d\U0001f52c|\U0001f469\U0001f3fe\u200d\U0001f52c|\U0001f469\U0001f3ff\u200d"
            r"\U0001f52c|\U0001f468\U0001f3fb\u200d\U0001f4bb|\U0001f468\U0001f3fc\u200d\U0001f4bb|\U0001f468\U0001f3fd"
            r"\u200d\U0001f4bb|\U0001f468\U0001f3fe\u200d\U0001f4bb|\U0001f468\U0001f3ff\u200d\U0001f4bb|\U0001f469"
            r"\U0001f3fb\u200d\U0001f4bb|\U0001f469\U0001f3fc\u200d\U0001f4bb|\U0001f469\U0001f3fd\u200d\U0001f4bb|"
            r"\U0001f469\U0001f3fe\u200d\U0001f4bb|\U0001f469\U0001f3ff\u200d\U0001f4bb|\U0001f468\U0001f3fb\u200d"
            r"\U0001f3a4|\U0001f468\U0001f3fc\u200d\U0001f3a4|\U0001f468\U0001f3fd\u200d\U0001f3a4|\U0001f468\U0001f3fe"
            r"\u200d\U0001f3a4|\U0001f468\U0001f3ff\u200d\U0001f3a4|\U0001f469\U0001f3fb\u200d\U0001f3a4|\U0001f469"
            r"\U0001f3fc\u200d\U0001f3a4|\U0001f469\U0001f3fd\u200d\U0001f3a4|\U0001f469\U0001f3fe\u200d\U0001f3a4|"
            r"\U0001f469\U0001f3ff\u200d\U0001f3a4|\U0001f468\U0001f3fb\u200d\U0001f3a8|\U0001f468\U0001f3fc\u200d"
            r"\U0001f3a8|\U0001f468\U0001f3fd\u200d\U0001f3a8|\U0001f468\U0001f3fe\u200d\U0001f3a8|\U0001f468\U0001f3ff"
            r"\u200d\U0001f3a8|\U0001f469\U0001f3fb\u200d\U0001f3a8|\U0001f469\U0001f3fc\u200d\U0001f3a8|\U0001f469"
            r"\U0001f3fd\u200d\U0001f3a8|\U0001f469\U0001f3fe\u200d\U0001f3a8|\U0001f469\U0001f3ff\u200d\U0001f3a8|"
            r"\U0001f468\u200d\u2708\ufe0f|\U0001f468\U0001f3fb\u200d\u2708|\U0001f468\U0001f3fc\u200d\u2708|\U0001f468"
            r"\U0001f3fd\u200d\u2708|\U0001f468\U0001f3fe\u200d\u2708|\U0001f468\U0001f3ff\u200d\u2708|\U0001f469\u200d"
            r"\u2708\ufe0f|\U0001f469\U0001f3fb\u200d\u2708|\U0001f469\U0001f3fc\u200d\u2708|\U0001f469\U0001f3fd\u200d"
            r"\u2708|\U0001f469\U0001f3fe\u200d\u2708|\U0001f469\U0001f3ff\u200d\u2708|\U0001f468\U0001f3fb\u200d"
            r"\U0001f680|\U0001f468\U0001f3fc\u200d\U0001f680|\U0001f468\U0001f3fd\u200d\U0001f680|\U0001f468\U0001f3fe"
            r"\u200d\U0001f680|\U0001f468\U0001f3ff\u200d\U0001f680|\U0001f469\U0001f3fb\u200d\U0001f680|\U0001f469"
            r"\U0001f3fc\u200d\U0001f680|\U0001f469\U0001f3fd\u200d\U0001f680|\U0001f469\U0001f3fe\u200d\U0001f680|"
            r"\U0001f469\U0001f3ff\u200d\U0001f680|\U0001f468\U0001f3fb\u200d\U0001f692|\U0001f468\U0001f3fc\u200d"
            r"\U0001f692|\U0001f468\U0001f3fd\u200d\U0001f692|\U0001f468\U0001f3fe\u200d\U0001f692|\U0001f468\U0001f3ff"
            r"\u200d\U0001f692|\U0001f469\U0001f3fb\u200d\U0001f692|\U0001f469\U0001f3fc\u200d\U0001f692|\U0001f469"
            r"\U0001f3fd\u200d\U0001f692|\U0001f469\U0001f3fe\u200d\U0001f692|\U0001f469\U0001f3ff\u200d\U0001f692|"
            r"\U0001f46e\u200d\u2642\ufe0f|\U0001f46e\U0001f3fb\u200d\u2642|\U0001f46e\U0001f3fc\u200d\u2642|\U0001f46e"
            r"\U0001f3fd\u200d\u2642|\U0001f46e\U0001f3fe\u200d\u2642|\U0001f46e\U0001f3ff\u200d\u2642|\U0001f46e\u200d"
            r"\u2640\ufe0f|\U0001f46e\U0001f3fb\u200d\u2640|\U0001f46e\U0001f3fc\u200d\u2640|\U0001f46e\U0001f3fd\u200d"
            r"\u2640|\U0001f46e\U0001f3fe\u200d\u2640|\U0001f46e\U0001f3ff\u200d\u2640|\U0001f575\u200d\u2642\ufe0f|"
            r"\U0001f575\ufe0f\u200d\u2642|\U0001f575\U0001f3fb\u200d\u2642|\U0001f575\U0001f3fc\u200d\u2642|\U0001f575"
            r"\U0001f3fd\u200d\u2642|\U0001f575\U0001f3fe\u200d\u2642|\U0001f575\U0001f3ff\u200d\u2642|\U0001f575\u200d"
            r"\u2640\ufe0f|\U0001f575\ufe0f\u200d\u2640|\U0001f575\U0001f3fb\u200d\u2640|\U0001f575\U0001f3fc\u200d"
            r"\u2640|\U0001f575\U0001f3fd\u200d\u2640|\U0001f575\U0001f3fe\u200d\u2640|\U0001f575\U0001f3ff\u200d"
            r"\u2640|\U0001f482\u200d\u2642\ufe0f|\U0001f482\U0001f3fb\u200d\u2642|\U0001f482\U0001f3fc\u200d\u2642|"
            r"\U0001f482\U0001f3fd\u200d\u2642|\U0001f482\U0001f3fe\u200d\u2642|\U0001f482\U0001f3ff\u200d\u2642|"
            r"\U0001f482\u200d\u2640\ufe0f|\U0001f482\U0001f3fb\u200d\u2640|\U0001f482\U0001f3fc\u200d\u2640|\U0001f482"
            r"\U0001f3fd\u200d\u2640|\U0001f482\U0001f3fe\u200d\u2640|\U0001f482\U0001f3ff\u200d\u2640|\U0001f477\u200d"
            r"\u2642\ufe0f|\U0001f477\U0001f3fb\u200d\u2642|\U0001f477\U0001f3fc\u200d\u2642|\U0001f477\U0001f3fd\u200d"
            r"\u2642|\U0001f477\U0001f3fe\u200d\u2642|\U0001f477\U0001f3ff\u200d\u2642|\U0001f477\u200d\u2640\ufe0f|"
            r"\U0001f477\U0001f3fb\u200d\u2640|\U0001f477\U0001f3fc\u200d\u2640|\U0001f477\U0001f3fd\u200d\u2640|"
            r"\U0001f477\U0001f3fe\u200d\u2640|\U0001f477\U0001f3ff\u200d\u2640|\U0001f473\u200d\u2642\ufe0f|\U0001f473"
            r"\U0001f3fb\u200d\u2642|\U0001f473\U0001f3fc\u200d\u2642|\U0001f473\U0001f3fd\u200d\u2642|\U0001f473"
            r"\U0001f3fe\u200d\u2642|\U0001f473\U0001f3ff\u200d\u2642|\U0001f473\u200d\u2640\ufe0f|\U0001f473\U0001f3fb"
            r"\u200d\u2640|\U0001f473\U0001f3fc\u200d\u2640|\U0001f473\U0001f3fd\u200d\u2640|\U0001f473\U0001f3fe\u200d"
            r"\u2640|\U0001f473\U0001f3ff\u200d\u2640|\U0001f9b8\u200d\u2642\ufe0f|\U0001f9b8\U0001f3fb\u200d\u2642|"
            r"\U0001f9b8\U0001f3fc\u200d\u2642|\U0001f9b8\U0001f3fd\u200d\u2642|\U0001f9b8\U0001f3fe\u200d\u2642|"
            r"\U0001f9b8\U0001f3ff\u200d\u2642|\U0001f9b8\u200d\u2640\ufe0f|\U0001f9b8\U0001f3fb\u200d\u2640|\U0001f9b8"
            r"\U0001f3fc\u200d\u2640|\U0001f9b8\U0001f3fd\u200d\u2640|\U0001f9b8\U0001f3fe\u200d\u2640|\U0001f9b8"
            r"\U0001f3ff\u200d\u2640|\U0001f9b9\u200d\u2642\ufe0f|\U0001f9b9\U0001f3fb\u200d\u2642|\U0001f9b9\U0001f3fc"
            r"\u200d\u2642|\U0001f9b9\U0001f3fd\u200d\u2642|\U0001f9b9\U0001f3fe\u200d\u2642|\U0001f9b9\U0001f3ff\u200d"
            r"\u2642|\U0001f9b9\u200d\u2640\ufe0f|\U0001f9b9\U0001f3fb\u200d\u2640|\U0001f9b9\U0001f3fc\u200d\u2640|"
            r"\U0001f9b9\U0001f3fd\u200d\u2640|\U0001f9b9\U0001f3fe\u200d\u2640|\U0001f9b9\U0001f3ff\u200d\u2640|"
            r"\U0001f9d9\u200d\u2642\ufe0f|\U0001f9d9\U0001f3fb\u200d\u2642|\U0001f9d9\U0001f3fc\u200d\u2642|\U0001f9d9"
            r"\U0001f3fd\u200d\u2642|\U0001f9d9\U0001f3fe\u200d\u2642|\U0001f9d9\U0001f3ff\u200d\u2642|\U0001f9d9\u200d"
            r"\u2640\ufe0f|\U0001f9d9\U0001f3fb\u200d\u2640|\U0001f9d9\U0001f3fc\u200d\u2640|\U0001f9d9\U0001f3fd\u200d"
            r"\u2640|\U0001f9d9\U0001f3fe\u200d\u2640|\U0001f9d9\U0001f3ff\u200d\u2640|\U0001f9da\u200d\u2642\ufe0f|"
            r"\U0001f9da\U0001f3fb\u200d\u2642|\U0001f9da\U0001f3fc\u200d\u2642|\U0001f9da\U0001f3fd\u200d\u2642|"
            r"\U0001f9da\U0001f3fe\u200d\u2642|\U0001f9da\U0001f3ff\u200d\u2642|\U0001f9da\u200d\u2640\ufe0f|\U0001f9da"
            r"\U0001f3fb\u200d\u2640|\U0001f9da\U0001f3fc\u200d\u2640|\U0001f9da\U0001f3fd\u200d\u2640|\U0001f9da"
            r"\U0001f3fe\u200d\u2640|\U0001f9da\U0001f3ff\u200d\u2640|\U0001f9db\u200d\u2642\ufe0f|\U0001f9db\U0001f3fb"
            r"\u200d\u2642|\U0001f9db\U0001f3fc\u200d\u2642|\U0001f9db\U0001f3fd\u200d\u2642|\U0001f9db\U0001f3fe\u200d"
            r"\u2642|\U0001f9db\U0001f3ff\u200d\u2642|\U0001f9db\u200d\u2640\ufe0f|\U0001f9db\U0001f3fb\u200d\u2640|"
            r"\U0001f9db\U0001f3fc\u200d\u2640|\U0001f9db\U0001f3fd\u200d\u2640|\U0001f9db\U0001f3fe\u200d\u2640|"
            r"\U0001f9db\U0001f3ff\u200d\u2640|\U0001f9dc\u200d\u2642\ufe0f|\U0001f9dc\U0001f3fb\u200d\u2642|\U0001f9dc"
            r"\U0001f3fc\u200d\u2642|\U0001f9dc\U0001f3fd\u200d\u2642|\U0001f9dc\U0001f3fe\u200d\u2642|\U0001f9dc"
            r"\U0001f3ff\u200d\u2642|\U0001f9dc\u200d\u2640\ufe0f|\U0001f9dc\U0001f3fb\u200d\u2640|\U0001f9dc\U0001f3fc"
            r"\u200d\u2640|\U0001f9dc\U0001f3fd\u200d\u2640|\U0001f9dc\U0001f3fe\u200d\u2640|\U0001f9dc\U0001f3ff\u200d"
            r"\u2640|\U0001f9dd\u200d\u2642\ufe0f|\U0001f9dd\U0001f3fb\u200d\u2642|\U0001f9dd\U0001f3fc\u200d\u2642|"
            r"\U0001f9dd\U0001f3fd\u200d\u2642|\U0001f9dd\U0001f3fe\u200d\u2642|\U0001f9dd\U0001f3ff\u200d\u2642|"
            r"\U0001f9dd\u200d\u2640\ufe0f|\U0001f9dd\U0001f3fb\u200d\u2640|\U0001f9dd\U0001f3fc\u200d\u2640|\U0001f9dd"
            r"\U0001f3fd\u200d\u2640|\U0001f9dd\U0001f3fe\u200d\u2640|\U0001f9dd\U0001f3ff\u200d\u2640|\U0001f9de\u200d"
            r"\u2642\ufe0f|\U0001f9de\u200d\u2640\ufe0f|\U0001f9df\u200d\u2642\ufe0f|\U0001f9df\u200d\u2640\ufe0f|"
            r"\U0001f486\u200d\u2642\ufe0f|\U0001f486\U0001f3fb\u200d\u2642|\U0001f486\U0001f3fc\u200d\u2642|\U0001f486"
            r"\U0001f3fd\u200d\u2642|\U0001f486\U0001f3fe\u200d\u2642|\U0001f486\U0001f3ff\u200d\u2642|\U0001f486\u200d"
            r"\u2640\ufe0f|\U0001f486\U0001f3fb\u200d\u2640|\U0001f486\U0001f3fc\u200d\u2640|\U0001f486\U0001f3fd\u200d"
            r"\u2640|\U0001f486\U0001f3fe\u200d\u2640|\U0001f486\U0001f3ff\u200d\u2640|\U0001f487\u200d\u2642\ufe0f|"
            r"\U0001f487\U0001f3fb\u200d\u2642|\U0001f487\U0001f3fc\u200d\u2642|\U0001f487\U0001f3fd\u200d\u2642|"
            r"\U0001f487\U0001f3fe\u200d\u2642|\U0001f487\U0001f3ff\u200d\u2642|\U0001f487\u200d\u2640\ufe0f|\U0001f487"
            r"\U0001f3fb\u200d\u2640|\U0001f487\U0001f3fc\u200d\u2640|\U0001f487\U0001f3fd\u200d\u2640|\U0001f487"
            r"\U0001f3fe\u200d\u2640|\U0001f487\U0001f3ff\u200d\u2640|\U0001f6b6\u200d\u2642\ufe0f|\U0001f6b6\U0001f3fb"
            r"\u200d\u2642|\U0001f6b6\U0001f3fc\u200d\u2642|\U0001f6b6\U0001f3fd\u200d\u2642|\U0001f6b6\U0001f3fe\u200d"
            r"\u2642|\U0001f6b6\U0001f3ff\u200d\u2642|\U0001f6b6\u200d\u2640\ufe0f|\U0001f6b6\U0001f3fb\u200d\u2640|"
            r"\U0001f6b6\U0001f3fc\u200d\u2640|\U0001f6b6\U0001f3fd\u200d\u2640|\U0001f6b6\U0001f3fe\u200d\u2640|"
            r"\U0001f6b6\U0001f3ff\u200d\u2640|\U0001f9cd\u200d\u2642\ufe0f|\U0001f9cd\U0001f3fb\u200d\u2642|\U0001f9cd"
            r"\U0001f3fc\u200d\u2642|\U0001f9cd\U0001f3fd\u200d\u2642|\U0001f9cd\U0001f3fe\u200d\u2642|\U0001f9cd"
            r"\U0001f3ff\u200d\u2642|\U0001f9cd\u200d\u2640\ufe0f|\U0001f9cd\U0001f3fb\u200d\u2640|\U0001f9cd\U0001f3fc"
            r"\u200d\u2640|\U0001f9cd\U0001f3fd\u200d\u2640|\U0001f9cd\U0001f3fe\u200d\u2640|\U0001f9cd\U0001f3ff\u200d"
            r"\u2640|\U0001f9ce\u200d\u2642\ufe0f|\U0001f9ce\U0001f3fb\u200d\u2642|\U0001f9ce\U0001f3fc\u200d\u2642|"
            r"\U0001f9ce\U0001f3fd\u200d\u2642|\U0001f9ce\U0001f3fe\u200d\u2642|\U0001f9ce\U0001f3ff\u200d\u2642|"
            r"\U0001f9ce\u200d\u2640\ufe0f|\U0001f9ce\U0001f3fb\u200d\u2640|\U0001f9ce\U0001f3fc\u200d\u2640|\U0001f9ce"
            r"\U0001f3fd\u200d\u2640|\U0001f9ce\U0001f3fe\u200d\u2640|\U0001f9ce\U0001f3ff\u200d\u2640|\U0001f468"
            r"\U0001f3fb\u200d\U0001f9af|\U0001f468\U0001f3fc\u200d\U0001f9af|\U0001f468\U0001f3fd\u200d\U0001f9af|"
            r"\U0001f468\U0001f3fe\u200d\U0001f9af|\U0001f468\U0001f3ff\u200d\U0001f9af|\U0001f469\U0001f3fb\u200d"
            r"\U0001f9af|\U0001f469\U0001f3fc\u200d\U0001f9af|\U0001f469\U0001f3fd\u200d\U0001f9af|\U0001f469\U0001f3fe"
            r"\u200d\U0001f9af|\U0001f469\U0001f3ff\u200d\U0001f9af|\U0001f468\U0001f3fb\u200d\U0001f9bc|\U0001f468"
            r"\U0001f3fc\u200d\U0001f9bc|\U0001f468\U0001f3fd\u200d\U0001f9bc|\U0001f468\U0001f3fe\u200d\U0001f9bc|"
            r"\U0001f468\U0001f3ff\u200d\U0001f9bc|\U0001f469\U0001f3fb\u200d\U0001f9bc|\U0001f469\U0001f3fc\u200d"
            r"\U0001f9bc|\U0001f469\U0001f3fd\u200d\U0001f9bc|\U0001f469\U0001f3fe\u200d\U0001f9bc|\U0001f469\U0001f3ff"
            r"\u200d\U0001f9bc|\U0001f468\U0001f3fb\u200d\U0001f9bd|\U0001f468\U0001f3fc\u200d\U0001f9bd|\U0001f468"
            r"\U0001f3fd\u200d\U0001f9bd|\U0001f468\U0001f3fe\u200d\U0001f9bd|\U0001f468\U0001f3ff\u200d\U0001f9bd|"
            r"\U0001f469\U0001f3fb\u200d\U0001f9bd|\U0001f469\U0001f3fc\u200d\U0001f9bd|\U0001f469\U0001f3fd\u200d"
            r"\U0001f9bd|\U0001f469\U0001f3fe\u200d\U0001f9bd|\U0001f469\U0001f3ff\u200d\U0001f9bd|\U0001f3c3\u200d"
            r"\u2642\ufe0f|\U0001f3c3\U0001f3fb\u200d\u2642|\U0001f3c3\U0001f3fc\u200d\u2642|\U0001f3c3\U0001f3fd\u200d"
            r"\u2642|\U0001f3c3\U0001f3fe\u200d\u2642|\U0001f3c3\U0001f3ff\u200d\u2642|\U0001f3c3\u200d\u2640\ufe0f|"
            r"\U0001f3c3\U0001f3fb\u200d\u2640|\U0001f3c3\U0001f3fc\u200d\u2640|\U0001f3c3\U0001f3fd\u200d\u2640|"
            r"\U0001f3c3\U0001f3fe\u200d\u2640|\U0001f3c3\U0001f3ff\u200d\u2640|\U0001f46f\u200d\u2642\ufe0f|\U0001f46f"
            r"\u200d\u2640\ufe0f|\U0001f9d6\u200d\u2642\ufe0f|\U0001f9d6\U0001f3fb\u200d\u2642|\U0001f9d6\U0001f3fc"
            r"\u200d\u2642|\U0001f9d6\U0001f3fd\u200d\u2642|\U0001f9d6\U0001f3fe\u200d\u2642|\U0001f9d6\U0001f3ff\u200d"
            r"\u2642|\U0001f9d6\u200d\u2640\ufe0f|\U0001f9d6\U0001f3fb\u200d\u2640|\U0001f9d6\U0001f3fc\u200d\u2640|"
            r"\U0001f9d6\U0001f3fd\u200d\u2640|\U0001f9d6\U0001f3fe\u200d\u2640|\U0001f9d6\U0001f3ff\u200d\u2640|"
            r"\U0001f9d7\u200d\u2642\ufe0f|\U0001f9d7\U0001f3fb\u200d\u2642|\U0001f9d7\U0001f3fc\u200d\u2642|\U0001f9d7"
            r"\U0001f3fd\u200d\u2642|\U0001f9d7\U0001f3fe\u200d\u2642|\U0001f9d7\U0001f3ff\u200d\u2642|\U0001f9d7\u200d"
            r"\u2640\ufe0f|\U0001f9d7\U0001f3fb\u200d\u2640|\U0001f9d7\U0001f3fc\u200d\u2640|\U0001f9d7\U0001f3fd\u200d"
            r"\u2640|\U0001f9d7\U0001f3fe\u200d\u2640|\U0001f9d7\U0001f3ff\u200d\u2640|\U0001f3cc\u200d\u2642\ufe0f|"
            r"\U0001f3cc\ufe0f\u200d\u2642|\U0001f3cc\U0001f3fb\u200d\u2642|\U0001f3cc\U0001f3fc\u200d\u2642|\U0001f3cc"
            r"\U0001f3fd\u200d\u2642|\U0001f3cc\U0001f3fe\u200d\u2642|\U0001f3cc\U0001f3ff\u200d\u2642|\U0001f3cc\u200d"
            r"\u2640\ufe0f|\U0001f3cc\ufe0f\u200d\u2640|\U0001f3cc\U0001f3fb\u200d\u2640|\U0001f3cc\U0001f3fc\u200d"
            r"\u2640|\U0001f3cc\U0001f3fd\u200d\u2640|\U0001f3cc\U0001f3fe\u200d\u2640|\U0001f3cc\U0001f3ff\u200d"
            r"\u2640|\U0001f3c4\u200d\u2642\ufe0f|\U0001f3c4\U0001f3fb\u200d\u2642|\U0001f3c4\U0001f3fc\u200d\u2642|"
            r"\U0001f3c4\U0001f3fd\u200d\u2642|\U0001f3c4\U0001f3fe\u200d\u2642|\U0001f3c4\U0001f3ff\u200d\u2642|"
            r"\U0001f3c4\u200d\u2640\ufe0f|\U0001f3c4\U0001f3fb\u200d\u2640|\U0001f3c4\U0001f3fc\u200d\u2640|\U0001f3c4"
            r"\U0001f3fd\u200d\u2640|\U0001f3c4\U0001f3fe\u200d\u2640|\U0001f3c4\U0001f3ff\u200d\u2640|\U0001f6a3\u200d"
            r"\u2642\ufe0f|\U0001f6a3\U0001f3fb\u200d\u2642|\U0001f6a3\U0001f3fc\u200d\u2642|\U0001f6a3\U0001f3fd\u200d"
            r"\u2642|\U0001f6a3\U0001f3fe\u200d\u2642|\U0001f6a3\U0001f3ff\u200d\u2642|\U0001f6a3\u200d\u2640\ufe0f|"
            r"\U0001f6a3\U0001f3fb\u200d\u2640|\U0001f6a3\U0001f3fc\u200d\u2640|\U0001f6a3\U0001f3fd\u200d\u2640|"
            r"\U0001f6a3\U0001f3fe\u200d\u2640|\U0001f6a3\U0001f3ff\u200d\u2640|\U0001f3ca\u200d\u2642\ufe0f|\U0001f3ca"
            r"\U0001f3fb\u200d\u2642|\U0001f3ca\U0001f3fc\u200d\u2642|\U0001f3ca\U0001f3fd\u200d\u2642|\U0001f3ca"
            r"\U0001f3fe\u200d\u2642|\U0001f3ca\U0001f3ff\u200d\u2642|\U0001f3ca\u200d\u2640\ufe0f|\U0001f3ca\U0001f3fb"
            r"\u200d\u2640|\U0001f3ca\U0001f3fc\u200d\u2640|\U0001f3ca\U0001f3fd\u200d\u2640|\U0001f3ca\U0001f3fe\u200d"
            r"\u2640|\U0001f3ca\U0001f3ff\u200d\u2640|\u26f9\u200d\u2642\ufe0f|\u26f9\ufe0f\u200d\u2642|\u26f9"
            r"\U0001f3fb\u200d\u2642|\u26f9\U0001f3fc\u200d\u2642|\u26f9\U0001f3fd\u200d\u2642|\u26f9\U0001f3fe\u200d"
            r"\u2642|\u26f9\U0001f3ff\u200d\u2642|\u26f9\u200d\u2640\ufe0f|\u26f9\ufe0f\u200d\u2640|\u26f9\U0001f3fb"
            r"\u200d\u2640|\u26f9\U0001f3fc\u200d\u2640|\u26f9\U0001f3fd\u200d\u2640|\u26f9\U0001f3fe\u200d\u2640|"
            r"\u26f9\U0001f3ff\u200d\u2640|\U0001f3cb\u200d\u2642\ufe0f|\U0001f3cb\ufe0f\u200d\u2642|\U0001f3cb"
            r"\U0001f3fb\u200d\u2642|\U0001f3cb\U0001f3fc\u200d\u2642|\U0001f3cb\U0001f3fd\u200d\u2642|\U0001f3cb"
            r"\U0001f3fe\u200d\u2642|\U0001f3cb\U0001f3ff\u200d\u2642|\U0001f3cb\u200d\u2640\ufe0f|\U0001f3cb\ufe0f"
            r"\u200d\u2640|\U0001f3cb\U0001f3fb\u200d\u2640|\U0001f3cb\U0001f3fc\u200d\u2640|\U0001f3cb\U0001f3fd\u200d"
            r"\u2640|\U0001f3cb\U0001f3fe\u200d\u2640|\U0001f3cb\U0001f3ff\u200d\u2640|\U0001f6b4\u200d\u2642\ufe0f|"
            r"\U0001f6b4\U0001f3fb\u200d\u2642|\U0001f6b4\U0001f3fc\u200d\u2642|\U0001f6b4\U0001f3fd\u200d\u2642|"
            r"\U0001f6b4\U0001f3fe\u200d\u2642|\U0001f6b4\U0001f3ff\u200d\u2642|\U0001f6b4\u200d\u2640\ufe0f|\U0001f6b4"
            r"\U0001f3fb\u200d\u2640|\U0001f6b4\U0001f3fc\u200d\u2640|\U0001f6b4\U0001f3fd\u200d\u2640|\U0001f6b4"
            r"\U0001f3fe\u200d\u2640|\U0001f6b4\U0001f3ff\u200d\u2640|\U0001f6b5\u200d\u2642\ufe0f|\U0001f6b5\U0001f3fb"
            r"\u200d\u2642|\U0001f6b5\U0001f3fc\u200d\u2642|\U0001f6b5\U0001f3fd\u200d\u2642|\U0001f6b5\U0001f3fe\u200d"
            r"\u2642|\U0001f6b5\U0001f3ff\u200d\u2642|\U0001f6b5\u200d\u2640\ufe0f|\U0001f6b5\U0001f3fb\u200d\u2640|"
            r"\U0001f6b5\U0001f3fc\u200d\u2640|\U0001f6b5\U0001f3fd\u200d\u2640|\U0001f6b5\U0001f3fe\u200d\u2640|"
            r"\U0001f6b5\U0001f3ff\u200d\u2640|\U0001f938\u200d\u2642\ufe0f|\U0001f938\U0001f3fb\u200d\u2642|\U0001f938"
            r"\U0001f3fc\u200d\u2642|\U0001f938\U0001f3fd\u200d\u2642|\U0001f938\U0001f3fe\u200d\u2642|\U0001f938"
            r"\U0001f3ff\u200d\u2642|\U0001f938\u200d\u2640\ufe0f|\U0001f938\U0001f3fb\u200d\u2640|\U0001f938\U0001f3fc"
            r"\u200d\u2640|\U0001f938\U0001f3fd\u200d\u2640|\U0001f938\U0001f3fe\u200d\u2640|\U0001f938\U0001f3ff\u200d"
            r"\u2640|\U0001f93c\u200d\u2642\ufe0f|\U0001f93c\u200d\u2640\ufe0f|\U0001f93d\u200d\u2642\ufe0f|\U0001f93d"
            r"\U0001f3fb\u200d\u2642|\U0001f93d\U0001f3fc\u200d\u2642|\U0001f93d\U0001f3fd\u200d\u2642|\U0001f93d"
            r"\U0001f3fe\u200d\u2642|\U0001f93d\U0001f3ff\u200d\u2642|\U0001f93d\u200d\u2640\ufe0f|\U0001f93d\U0001f3fb"
            r"\u200d\u2640|\U0001f93d\U0001f3fc\u200d\u2640|\U0001f93d\U0001f3fd\u200d\u2640|\U0001f93d\U0001f3fe\u200d"
            r"\u2640|\U0001f93d\U0001f3ff\u200d\u2640|\U0001f93e\u200d\u2642\ufe0f|\U0001f93e\U0001f3fb\u200d\u2642|"
            r"\U0001f93e\U0001f3fc\u200d\u2642|\U0001f93e\U0001f3fd\u200d\u2642|\U0001f93e\U0001f3fe\u200d\u2642|"
            r"\U0001f93e\U0001f3ff\u200d\u2642|\U0001f93e\u200d\u2640\ufe0f|\U0001f93e\U0001f3fb\u200d\u2640|\U0001f93e"
            r"\U0001f3fc\u200d\u2640|\U0001f93e\U0001f3fd\u200d\u2640|\U0001f93e\U0001f3fe\u200d\u2640|\U0001f93e"
            r"\U0001f3ff\u200d\u2640|\U0001f939\u200d\u2642\ufe0f|\U0001f939\U0001f3fb\u200d\u2642|\U0001f939\U0001f3fc"
            r"\u200d\u2642|\U0001f939\U0001f3fd\u200d\u2642|\U0001f939\U0001f3fe\u200d\u2642|\U0001f939\U0001f3ff\u200d"
            r"\u2642|\U0001f939\u200d\u2640\ufe0f|\U0001f939\U0001f3fb\u200d\u2640|\U0001f939\U0001f3fc\u200d\u2640|"
            r"\U0001f939\U0001f3fd\u200d\u2640|\U0001f939\U0001f3fe\u200d\u2640|\U0001f939\U0001f3ff\u200d\u2640|"
            r"\U0001f9d8\u200d\u2642\ufe0f|\U0001f9d8\U0001f3fb\u200d\u2642|\U0001f9d8\U0001f3fc\u200d\u2642|\U0001f9d8"
            r"\U0001f3fd\u200d\u2642|\U0001f9d8\U0001f3fe\u200d\u2642|\U0001f9d8\U0001f3ff\u200d\u2642|\U0001f9d8\u200d"
            r"\u2640\ufe0f|\U0001f9d8\U0001f3fb\u200d\u2640|\U0001f9d8\U0001f3fc\u200d\u2640|\U0001f9d8\U0001f3fd\u200d"
            r"\u2640|\U0001f9d8\U0001f3fe\u200d\u2640|\U0001f9d8\U0001f3ff\u200d\u2640|\U0001f3f3\ufe0f\u200d"
            r"\U0001f308|\U0001f3f4\u200d\u2620\ufe0f|\U0001f441\u200d\U0001f5e8|\U0001f471\u200d\u2642|\U0001f468"
            r"\u200d\U0001f9b0|\U0001f468\u200d\U0001f9b1|\U0001f468\u200d\U0001f9b3|\U0001f468\u200d\U0001f9b2|"
            r"\U0001f471\u200d\u2640|\U0001f469\u200d\U0001f9b0|\U0001f469\u200d\U0001f9b1|\U0001f469\u200d\U0001f9b3|"
            r"\U0001f469\u200d\U0001f9b2|\U0001f64d\u200d\u2642|\U0001f64d\u200d\u2640|\U0001f64e\u200d\u2642|"
            r"\U0001f64e\u200d\u2640|\U0001f645\u200d\u2642|\U0001f645\u200d\u2640|\U0001f646\u200d\u2642|\U0001f646"
            r"\u200d\u2640|\U0001f481\u200d\u2642|\U0001f481\u200d\u2640|\U0001f64b\u200d\u2642|\U0001f64b\u200d\u2640|"
            r"\U0001f9cf\u200d\u2642|\U0001f9cf\u200d\u2640|\U0001f647\u200d\u2642|\U0001f647\u200d\u2640|\U0001f926"
            r"\u200d\u2642|\U0001f926\u200d\u2640|\U0001f937\u200d\u2642|\U0001f937\u200d\u2640|\U0001f468\u200d\u2695|"
            r"\U0001f469\u200d\u2695|\U0001f468\u200d\U0001f393|\U0001f469\u200d\U0001f393|\U0001f468\u200d\U0001f3eb|"
            r"\U0001f469\u200d\U0001f3eb|\U0001f468\u200d\u2696|\U0001f469\u200d\u2696|\U0001f468\u200d\U0001f33e|"
            r"\U0001f469\u200d\U0001f33e|\U0001f468\u200d\U0001f373|\U0001f469\u200d\U0001f373|\U0001f468\u200d"
            r"\U0001f527|\U0001f469\u200d\U0001f527|\U0001f468\u200d\U0001f3ed|\U0001f469\u200d\U0001f3ed|\U0001f468"
            r"\u200d\U0001f4bc|\U0001f469\u200d\U0001f4bc|\U0001f468\u200d\U0001f52c|\U0001f469\u200d\U0001f52c|"
            r"\U0001f468\u200d\U0001f4bb|\U0001f469\u200d\U0001f4bb|\U0001f468\u200d\U0001f3a4|\U0001f469\u200d"
            r"\U0001f3a4|\U0001f468\u200d\U0001f3a8|\U0001f469\u200d\U0001f3a8|\U0001f468\u200d\u2708|\U0001f469\u200d"
            r"\u2708|\U0001f468\u200d\U0001f680|\U0001f469\u200d\U0001f680|\U0001f468\u200d\U0001f692|\U0001f469\u200d"
            r"\U0001f692|\U0001f46e\u200d\u2642|\U0001f46e\u200d\u2640|\U0001f575\u200d\u2642|\U0001f575\u200d\u2640|"
            r"\U0001f482\u200d\u2642|\U0001f482\u200d\u2640|\U0001f477\u200d\u2642|\U0001f477\u200d\u2640|\U0001f473"
            r"\u200d\u2642|\U0001f473\u200d\u2640|\U0001f9b8\u200d\u2642|\U0001f9b8\u200d\u2640|\U0001f9b9\u200d\u2642|"
            r"\U0001f9b9\u200d\u2640|\U0001f9d9\u200d\u2642|\U0001f9d9\u200d\u2640|\U0001f9da\u200d\u2642|\U0001f9da"
            r"\u200d\u2640|\U0001f9db\u200d\u2642|\U0001f9db\u200d\u2640|\U0001f9dc\u200d\u2642|\U0001f9dc\u200d\u2640|"
            r"\U0001f9dd\u200d\u2642|\U0001f9dd\u200d\u2640|\U0001f9de\u200d\u2642|\U0001f9de\u200d\u2640|\U0001f9df"
            r"\u200d\u2642|\U0001f9df\u200d\u2640|\U0001f486\u200d\u2642|\U0001f486\u200d\u2640|\U0001f487\u200d\u2642|"
            r"\U0001f487\u200d\u2640|\U0001f6b6\u200d\u2642|\U0001f6b6\u200d\u2640|\U0001f9cd\u200d\u2642|\U0001f9cd"
            r"\u200d\u2640|\U0001f9ce\u200d\u2642|\U0001f9ce\u200d\u2640|\U0001f468\u200d\U0001f9af|\U0001f469\u200d"
            r"\U0001f9af|\U0001f468\u200d\U0001f9bc|\U0001f469\u200d\U0001f9bc|\U0001f468\u200d\U0001f9bd|\U0001f469"
            r"\u200d\U0001f9bd|\U0001f3c3\u200d\u2642|\U0001f3c3\u200d\u2640|\U0001f46f\u200d\u2642|\U0001f46f\u200d"
            r"\u2640|\U0001f9d6\u200d\u2642|\U0001f9d6\u200d\u2640|\U0001f9d7\u200d\u2642|\U0001f9d7\u200d\u2640|"
            r"\U0001f3cc\u200d\u2642|\U0001f3cc\u200d\u2640|\U0001f3c4\u200d\u2642|\U0001f3c4\u200d\u2640|\U0001f6a3"
            r"\u200d\u2642|\U0001f6a3\u200d\u2640|\U0001f3ca\u200d\u2642|\U0001f3ca\u200d\u2640|\u26f9\u200d\u2642|"
            r"\u26f9\u200d\u2640|\U0001f3cb\u200d\u2642|\U0001f3cb\u200d\u2640|\U0001f6b4\u200d\u2642|\U0001f6b4\u200d"
            r"\u2640|\U0001f6b5\u200d\u2642|\U0001f6b5\u200d\u2640|\U0001f938\u200d\u2642|\U0001f938\u200d\u2640|"
            r"\U0001f93c\u200d\u2642|\U0001f93c\u200d\u2640|\U0001f93d\u200d\u2642|\U0001f93d\u200d\u2640|\U0001f93e"
            r"\u200d\u2642|\U0001f93e\u200d\u2640|\U0001f939\u200d\u2642|\U0001f939\u200d\u2640|\U0001f9d8\u200d\u2642|"
            r"\U0001f9d8\u200d\u2640|\U0001f468\u200d\U0001f466|\U0001f468\u200d\U0001f467|\U0001f469\u200d\U0001f466|"
            r"\U0001f469\u200d\U0001f467|\U0001f415\u200d\U0001f9ba|\\#\ufe0f\u20e3|\\*\ufe0f\u20e3|0\ufe0f\u20e3|1"
            r"\ufe0f\u20e3|2\ufe0f\u20e3|3\ufe0f\u20e3|4\ufe0f\u20e3|5\ufe0f\u20e3|6\ufe0f\u20e3|7\ufe0f\u20e3|8\ufe0f"
            r"\u20e3|9\ufe0f\u20e3|\U0001f3f3\u200d\U0001f308|\U0001f3f4\u200d\u2620|\u263a\ufe0f|\u2639\ufe0f|\u2620"
            r"\ufe0f|\u2763\ufe0f|\u2764\ufe0f|\U0001f573\ufe0f|\U0001f5e8\ufe0f|\U0001f5ef\ufe0f|\U0001f44b\U0001f3fb|"
            r"\U0001f44b\U0001f3fc|\U0001f44b\U0001f3fd|\U0001f44b\U0001f3fe|\U0001f44b\U0001f3ff|\U0001f91a\U0001f3fb|"
            r"\U0001f91a\U0001f3fc|\U0001f91a\U0001f3fd|\U0001f91a\U0001f3fe|\U0001f91a\U0001f3ff|\U0001f590\ufe0f|"
            r"\U0001f590\U0001f3fb|\U0001f590\U0001f3fc|\U0001f590\U0001f3fd|\U0001f590\U0001f3fe|\U0001f590\U0001f3ff|"
            r"\u270b\U0001f3fb|\u270b\U0001f3fc|\u270b\U0001f3fd|\u270b\U0001f3fe|\u270b\U0001f3ff|\U0001f596"
            r"\U0001f3fb|\U0001f596\U0001f3fc|\U0001f596\U0001f3fd|\U0001f596\U0001f3fe|\U0001f596\U0001f3ff|\U0001f44c"
            r"\U0001f3fb|\U0001f44c\U0001f3fc|\U0001f44c\U0001f3fd|\U0001f44c\U0001f3fe|\U0001f44c\U0001f3ff|\U0001f90f"
            r"\U0001f3fb|\U0001f90f\U0001f3fc|\U0001f90f\U0001f3fd|\U0001f90f\U0001f3fe|\U0001f90f\U0001f3ff|\u270c"
            r"\ufe0f|\u270c\U0001f3fb|\u270c\U0001f3fc|\u270c\U0001f3fd|\u270c\U0001f3fe|\u270c\U0001f3ff|\U0001f91e"
            r"\U0001f3fb|\U0001f91e\U0001f3fc|\U0001f91e\U0001f3fd|\U0001f91e\U0001f3fe|\U0001f91e\U0001f3ff|\U0001f91f"
            r"\U0001f3fb|\U0001f91f\U0001f3fc|\U0001f91f\U0001f3fd|\U0001f91f\U0001f3fe|\U0001f91f\U0001f3ff|\U0001f918"
            r"\U0001f3fb|\U0001f918\U0001f3fc|\U0001f918\U0001f3fd|\U0001f918\U0001f3fe|\U0001f918\U0001f3ff|\U0001f919"
            r"\U0001f3fb|\U0001f919\U0001f3fc|\U0001f919\U0001f3fd|\U0001f919\U0001f3fe|\U0001f919\U0001f3ff|\U0001f448"
            r"\U0001f3fb|\U0001f448\U0001f3fc|\U0001f448\U0001f3fd|\U0001f448\U0001f3fe|\U0001f448\U0001f3ff|\U0001f449"
            r"\U0001f3fb|\U0001f449\U0001f3fc|\U0001f449\U0001f3fd|\U0001f449\U0001f3fe|\U0001f449\U0001f3ff|\U0001f446"
            r"\U0001f3fb|\U0001f446\U0001f3fc|\U0001f446\U0001f3fd|\U0001f446\U0001f3fe|\U0001f446\U0001f3ff|\U0001f595"
            r"\U0001f3fb|\U0001f595\U0001f3fc|\U0001f595\U0001f3fd|\U0001f595\U0001f3fe|\U0001f595\U0001f3ff|\U0001f447"
            r"\U0001f3fb|\U0001f447\U0001f3fc|\U0001f447\U0001f3fd|\U0001f447\U0001f3fe|\U0001f447\U0001f3ff|\u261d"
            r"\ufe0f|\u261d\U0001f3fb|\u261d\U0001f3fc|\u261d\U0001f3fd|\u261d\U0001f3fe|\u261d\U0001f3ff|\U0001f44d"
            r"\U0001f3fb|\U0001f44d\U0001f3fc|\U0001f44d\U0001f3fd|\U0001f44d\U0001f3fe|\U0001f44d\U0001f3ff|\U0001f44e"
            r"\U0001f3fb|\U0001f44e\U0001f3fc|\U0001f44e\U0001f3fd|\U0001f44e\U0001f3fe|\U0001f44e\U0001f3ff|\u270a"
            r"\U0001f3fb|\u270a\U0001f3fc|\u270a\U0001f3fd|\u270a\U0001f3fe|\u270a\U0001f3ff|\U0001f44a\U0001f3fb|"
            r"\U0001f44a\U0001f3fc|\U0001f44a\U0001f3fd|\U0001f44a\U0001f3fe|\U0001f44a\U0001f3ff|\U0001f91b\U0001f3fb|"
            r"\U0001f91b\U0001f3fc|\U0001f91b\U0001f3fd|\U0001f91b\U0001f3fe|\U0001f91b\U0001f3ff|\U0001f91c\U0001f3fb|"
            r"\U0001f91c\U0001f3fc|\U0001f91c\U0001f3fd|\U0001f91c\U0001f3fe|\U0001f91c\U0001f3ff|\U0001f44f\U0001f3fb|"
            r"\U0001f44f\U0001f3fc|\U0001f44f\U0001f3fd|\U0001f44f\U0001f3fe|\U0001f44f\U0001f3ff|\U0001f64c\U0001f3fb|"
            r"\U0001f64c\U0001f3fc|\U0001f64c\U0001f3fd|\U0001f64c\U0001f3fe|\U0001f64c\U0001f3ff|\U0001f450\U0001f3fb|"
            r"\U0001f450\U0001f3fc|\U0001f450\U0001f3fd|\U0001f450\U0001f3fe|\U0001f450\U0001f3ff|\U0001f932\U0001f3fb|"
            r"\U0001f932\U0001f3fc|\U0001f932\U0001f3fd|\U0001f932\U0001f3fe|\U0001f932\U0001f3ff|\U0001f64f\U0001f3fb|"
            r"\U0001f64f\U0001f3fc|\U0001f64f\U0001f3fd|\U0001f64f\U0001f3fe|\U0001f64f\U0001f3ff|\u270d\ufe0f|\u270d"
            r"\U0001f3fb|\u270d\U0001f3fc|\u270d\U0001f3fd|\u270d\U0001f3fe|\u270d\U0001f3ff|\U0001f485\U0001f3fb|"
            r"\U0001f485\U0001f3fc|\U0001f485\U0001f3fd|\U0001f485\U0001f3fe|\U0001f485\U0001f3ff|\U0001f933\U0001f3fb|"
            r"\U0001f933\U0001f3fc|\U0001f933\U0001f3fd|\U0001f933\U0001f3fe|\U0001f933\U0001f3ff|\U0001f4aa\U0001f3fb|"
            r"\U0001f4aa\U0001f3fc|\U0001f4aa\U0001f3fd|\U0001f4aa\U0001f3fe|\U0001f4aa\U0001f3ff|\U0001f9b5\U0001f3fb|"
            r"\U0001f9b5\U0001f3fc|\U0001f9b5\U0001f3fd|\U0001f9b5\U0001f3fe|\U0001f9b5\U0001f3ff|\U0001f9b6\U0001f3fb|"
            r"\U0001f9b6\U0001f3fc|\U0001f9b6\U0001f3fd|\U0001f9b6\U0001f3fe|\U0001f9b6\U0001f3ff|\U0001f442\U0001f3fb|"
            r"\U0001f442\U0001f3fc|\U0001f442\U0001f3fd|\U0001f442\U0001f3fe|\U0001f442\U0001f3ff|\U0001f9bb\U0001f3fb|"
            r"\U0001f9bb\U0001f3fc|\U0001f9bb\U0001f3fd|\U0001f9bb\U0001f3fe|\U0001f9bb\U0001f3ff|\U0001f443\U0001f3fb|"
            r"\U0001f443\U0001f3fc|\U0001f443\U0001f3fd|\U0001f443\U0001f3fe|\U0001f443\U0001f3ff|\U0001f441\ufe0f|"
            r"\U0001f476\U0001f3fb|\U0001f476\U0001f3fc|\U0001f476\U0001f3fd|\U0001f476\U0001f3fe|\U0001f476\U0001f3ff|"
            r"\U0001f9d2\U0001f3fb|\U0001f9d2\U0001f3fc|\U0001f9d2\U0001f3fd|\U0001f9d2\U0001f3fe|\U0001f9d2\U0001f3ff|"
            r"\U0001f466\U0001f3fb|\U0001f466\U0001f3fc|\U0001f466\U0001f3fd|\U0001f466\U0001f3fe|\U0001f466\U0001f3ff|"
            r"\U0001f467\U0001f3fb|\U0001f467\U0001f3fc|\U0001f467\U0001f3fd|\U0001f467\U0001f3fe|\U0001f467\U0001f3ff|"
            r"\U0001f9d1\U0001f3fb|\U0001f9d1\U0001f3fc|\U0001f9d1\U0001f3fd|\U0001f9d1\U0001f3fe|\U0001f9d1\U0001f3ff|"
            r"\U0001f471\U0001f3fb|\U0001f471\U0001f3fc|\U0001f471\U0001f3fd|\U0001f471\U0001f3fe|\U0001f471\U0001f3ff|"
            r"\U0001f468\U0001f3fb|\U0001f468\U0001f3fc|\U0001f468\U0001f3fd|\U0001f468\U0001f3fe|\U0001f468\U0001f3ff|"
            r"\U0001f9d4\U0001f3fb|\U0001f9d4\U0001f3fc|\U0001f9d4\U0001f3fd|\U0001f9d4\U0001f3fe|\U0001f9d4\U0001f3ff|"
            r"\U0001f469\U0001f3fb|\U0001f469\U0001f3fc|\U0001f469\U0001f3fd|\U0001f469\U0001f3fe|\U0001f469\U0001f3ff|"
            r"\U0001f9d3\U0001f3fb|\U0001f9d3\U0001f3fc|\U0001f9d3\U0001f3fd|\U0001f9d3\U0001f3fe|\U0001f9d3\U0001f3ff|"
            r"\U0001f474\U0001f3fb|\U0001f474\U0001f3fc|\U0001f474\U0001f3fd|\U0001f474\U0001f3fe|\U0001f474\U0001f3ff|"
            r"\U0001f475\U0001f3fb|\U0001f475\U0001f3fc|\U0001f475\U0001f3fd|\U0001f475\U0001f3fe|\U0001f475\U0001f3ff|"
            r"\U0001f64d\U0001f3fb|\U0001f64d\U0001f3fc|\U0001f64d\U0001f3fd|\U0001f64d\U0001f3fe|\U0001f64d\U0001f3ff|"
            r"\U0001f64e\U0001f3fb|\U0001f64e\U0001f3fc|\U0001f64e\U0001f3fd|\U0001f64e\U0001f3fe|\U0001f64e\U0001f3ff|"
            r"\U0001f645\U0001f3fb|\U0001f645\U0001f3fc|\U0001f645\U0001f3fd|\U0001f645\U0001f3fe|\U0001f645\U0001f3ff|"
            r"\U0001f646\U0001f3fb|\U0001f646\U0001f3fc|\U0001f646\U0001f3fd|\U0001f646\U0001f3fe|\U0001f646\U0001f3ff|"
            r"\U0001f481\U0001f3fb|\U0001f481\U0001f3fc|\U0001f481\U0001f3fd|\U0001f481\U0001f3fe|\U0001f481\U0001f3ff|"
            r"\U0001f64b\U0001f3fb|\U0001f64b\U0001f3fc|\U0001f64b\U0001f3fd|\U0001f64b\U0001f3fe|\U0001f64b\U0001f3ff|"
            r"\U0001f9cf\U0001f3fb|\U0001f9cf\U0001f3fc|\U0001f9cf\U0001f3fd|\U0001f9cf\U0001f3fe|\U0001f9cf\U0001f3ff|"
            r"\U0001f647\U0001f3fb|\U0001f647\U0001f3fc|\U0001f647\U0001f3fd|\U0001f647\U0001f3fe|\U0001f647\U0001f3ff|"
            r"\U0001f926\U0001f3fb|\U0001f926\U0001f3fc|\U0001f926\U0001f3fd|\U0001f926\U0001f3fe|\U0001f926\U0001f3ff|"
            r"\U0001f937\U0001f3fb|\U0001f937\U0001f3fc|\U0001f937\U0001f3fd|\U0001f937\U0001f3fe|\U0001f937\U0001f3ff|"
            r"\U0001f46e\U0001f3fb|\U0001f46e\U0001f3fc|\U0001f46e\U0001f3fd|\U0001f46e\U0001f3fe|\U0001f46e\U0001f3ff|"
            r"\U0001f575\ufe0f|\U0001f575\U0001f3fb|\U0001f575\U0001f3fc|\U0001f575\U0001f3fd|\U0001f575\U0001f3fe|"
            r"\U0001f575\U0001f3ff|\U0001f482\U0001f3fb|\U0001f482\U0001f3fc|\U0001f482\U0001f3fd|\U0001f482\U0001f3fe|"
            r"\U0001f482\U0001f3ff|\U0001f477\U0001f3fb|\U0001f477\U0001f3fc|\U0001f477\U0001f3fd|\U0001f477\U0001f3fe|"
            r"\U0001f477\U0001f3ff|\U0001f934\U0001f3fb|\U0001f934\U0001f3fc|\U0001f934\U0001f3fd|\U0001f934\U0001f3fe|"
            r"\U0001f934\U0001f3ff|\U0001f478\U0001f3fb|\U0001f478\U0001f3fc|\U0001f478\U0001f3fd|\U0001f478\U0001f3fe|"
            r"\U0001f478\U0001f3ff|\U0001f473\U0001f3fb|\U0001f473\U0001f3fc|\U0001f473\U0001f3fd|\U0001f473\U0001f3fe|"
            r"\U0001f473\U0001f3ff|\U0001f472\U0001f3fb|\U0001f472\U0001f3fc|\U0001f472\U0001f3fd|\U0001f472\U0001f3fe|"
            r"\U0001f472\U0001f3ff|\U0001f9d5\U0001f3fb|\U0001f9d5\U0001f3fc|\U0001f9d5\U0001f3fd|\U0001f9d5\U0001f3fe|"
            r"\U0001f9d5\U0001f3ff|\U0001f935\U0001f3fb|\U0001f935\U0001f3fc|\U0001f935\U0001f3fd|\U0001f935\U0001f3fe|"
            r"\U0001f935\U0001f3ff|\U0001f470\U0001f3fb|\U0001f470\U0001f3fc|\U0001f470\U0001f3fd|\U0001f470\U0001f3fe|"
            r"\U0001f470\U0001f3ff|\U0001f930\U0001f3fb|\U0001f930\U0001f3fc|\U0001f930\U0001f3fd|\U0001f930\U0001f3fe|"
            r"\U0001f930\U0001f3ff|\U0001f931\U0001f3fb|\U0001f931\U0001f3fc|\U0001f931\U0001f3fd|\U0001f931\U0001f3fe|"
            r"\U0001f931\U0001f3ff|\U0001f47c\U0001f3fb|\U0001f47c\U0001f3fc|\U0001f47c\U0001f3fd|\U0001f47c\U0001f3fe|"
            r"\U0001f47c\U0001f3ff|\U0001f385\U0001f3fb|\U0001f385\U0001f3fc|\U0001f385\U0001f3fd|\U0001f385\U0001f3fe|"
            r"\U0001f385\U0001f3ff|\U0001f936\U0001f3fb|\U0001f936\U0001f3fc|\U0001f936\U0001f3fd|\U0001f936\U0001f3fe|"
            r"\U0001f936\U0001f3ff|\U0001f9b8\U0001f3fb|\U0001f9b8\U0001f3fc|\U0001f9b8\U0001f3fd|\U0001f9b8\U0001f3fe|"
            r"\U0001f9b8\U0001f3ff|\U0001f9b9\U0001f3fb|\U0001f9b9\U0001f3fc|\U0001f9b9\U0001f3fd|\U0001f9b9\U0001f3fe|"
            r"\U0001f9b9\U0001f3ff|\U0001f9d9\U0001f3fb|\U0001f9d9\U0001f3fc|\U0001f9d9\U0001f3fd|\U0001f9d9\U0001f3fe|"
            r"\U0001f9d9\U0001f3ff|\U0001f9da\U0001f3fb|\U0001f9da\U0001f3fc|\U0001f9da\U0001f3fd|\U0001f9da\U0001f3fe|"
            r"\U0001f9da\U0001f3ff|\U0001f9db\U0001f3fb|\U0001f9db\U0001f3fc|\U0001f9db\U0001f3fd|\U0001f9db\U0001f3fe|"
            r"\U0001f9db\U0001f3ff|\U0001f9dc\U0001f3fb|\U0001f9dc\U0001f3fc|\U0001f9dc\U0001f3fd|\U0001f9dc\U0001f3fe|"
            r"\U0001f9dc\U0001f3ff|\U0001f9dd\U0001f3fb|\U0001f9dd\U0001f3fc|\U0001f9dd\U0001f3fd|\U0001f9dd\U0001f3fe|"
            r"\U0001f9dd\U0001f3ff|\U0001f486\U0001f3fb|\U0001f486\U0001f3fc|\U0001f486\U0001f3fd|\U0001f486\U0001f3fe|"
            r"\U0001f486\U0001f3ff|\U0001f487\U0001f3fb|\U0001f487\U0001f3fc|\U0001f487\U0001f3fd|\U0001f487\U0001f3fe|"
            r"\U0001f487\U0001f3ff|\U0001f6b6\U0001f3fb|\U0001f6b6\U0001f3fc|\U0001f6b6\U0001f3fd|\U0001f6b6\U0001f3fe|"
            r"\U0001f6b6\U0001f3ff|\U0001f9cd\U0001f3fb|\U0001f9cd\U0001f3fc|\U0001f9cd\U0001f3fd|\U0001f9cd\U0001f3fe|"
            r"\U0001f9cd\U0001f3ff|\U0001f9ce\U0001f3fb|\U0001f9ce\U0001f3fc|\U0001f9ce\U0001f3fd|\U0001f9ce\U0001f3fe|"
            r"\U0001f9ce\U0001f3ff|\U0001f3c3\U0001f3fb|\U0001f3c3\U0001f3fc|\U0001f3c3\U0001f3fd|\U0001f3c3\U0001f3fe|"
            r"\U0001f3c3\U0001f3ff|\U0001f483\U0001f3fb|\U0001f483\U0001f3fc|\U0001f483\U0001f3fd|\U0001f483\U0001f3fe|"
            r"\U0001f483\U0001f3ff|\U0001f57a\U0001f3fb|\U0001f57a\U0001f3fc|\U0001f57a\U0001f3fd|\U0001f57a\U0001f3fe|"
            r"\U0001f57a\U0001f3ff|\U0001f574\ufe0f|\U0001f574\U0001f3fb|\U0001f574\U0001f3fc|\U0001f574\U0001f3fd|"
            r"\U0001f574\U0001f3fe|\U0001f574\U0001f3ff|\U0001f9d6\U0001f3fb|\U0001f9d6\U0001f3fc|\U0001f9d6\U0001f3fd|"
            r"\U0001f9d6\U0001f3fe|\U0001f9d6\U0001f3ff|\U0001f9d7\U0001f3fb|\U0001f9d7\U0001f3fc|\U0001f9d7\U0001f3fd|"
            r"\U0001f9d7\U0001f3fe|\U0001f9d7\U0001f3ff|\U0001f3c7\U0001f3fb|\U0001f3c7\U0001f3fc|\U0001f3c7\U0001f3fd|"
            r"\U0001f3c7\U0001f3fe|\U0001f3c7\U0001f3ff|\u26f7\ufe0f|\U0001f3c2\U0001f3fb|\U0001f3c2\U0001f3fc|"
            r"\U0001f3c2\U0001f3fd|\U0001f3c2\U0001f3fe|\U0001f3c2\U0001f3ff|\U0001f3cc\ufe0f|\U0001f3cc\U0001f3fb|"
            r"\U0001f3cc\U0001f3fc|\U0001f3cc\U0001f3fd|\U0001f3cc\U0001f3fe|\U0001f3cc\U0001f3ff|\U0001f3c4\U0001f3fb|"
            r"\U0001f3c4\U0001f3fc|\U0001f3c4\U0001f3fd|\U0001f3c4\U0001f3fe|\U0001f3c4\U0001f3ff|\U0001f6a3\U0001f3fb|"
            r"\U0001f6a3\U0001f3fc|\U0001f6a3\U0001f3fd|\U0001f6a3\U0001f3fe|\U0001f6a3\U0001f3ff|\U0001f3ca\U0001f3fb|"
            r"\U0001f3ca\U0001f3fc|\U0001f3ca\U0001f3fd|\U0001f3ca\U0001f3fe|\U0001f3ca\U0001f3ff|\u26f9\ufe0f|\u26f9"
            r"\U0001f3fb|\u26f9\U0001f3fc|\u26f9\U0001f3fd|\u26f9\U0001f3fe|\u26f9\U0001f3ff|\U0001f3cb\ufe0f|"
            r"\U0001f3cb\U0001f3fb|\U0001f3cb\U0001f3fc|\U0001f3cb\U0001f3fd|\U0001f3cb\U0001f3fe|\U0001f3cb\U0001f3ff|"
            r"\U0001f6b4\U0001f3fb|\U0001f6b4\U0001f3fc|\U0001f6b4\U0001f3fd|\U0001f6b4\U0001f3fe|\U0001f6b4\U0001f3ff|"
            r"\U0001f6b5\U0001f3fb|\U0001f6b5\U0001f3fc|\U0001f6b5\U0001f3fd|\U0001f6b5\U0001f3fe|\U0001f6b5\U0001f3ff|"
            r"\U0001f938\U0001f3fb|\U0001f938\U0001f3fc|\U0001f938\U0001f3fd|\U0001f938\U0001f3fe|\U0001f938\U0001f3ff|"
            r"\U0001f93d\U0001f3fb|\U0001f93d\U0001f3fc|\U0001f93d\U0001f3fd|\U0001f93d\U0001f3fe|\U0001f93d\U0001f3ff|"
            r"\U0001f93e\U0001f3fb|\U0001f93e\U0001f3fc|\U0001f93e\U0001f3fd|\U0001f93e\U0001f3fe|\U0001f93e\U0001f3ff|"
            r"\U0001f939\U0001f3fb|\U0001f939\U0001f3fc|\U0001f939\U0001f3fd|\U0001f939\U0001f3fe|\U0001f939\U0001f3ff|"
            r"\U0001f9d8\U0001f3fb|\U0001f9d8\U0001f3fc|\U0001f9d8\U0001f3fd|\U0001f9d8\U0001f3fe|\U0001f9d8\U0001f3ff|"
            r"\U0001f6c0\U0001f3fb|\U0001f6c0\U0001f3fc|\U0001f6c0\U0001f3fd|\U0001f6c0\U0001f3fe|\U0001f6c0\U0001f3ff|"
            r"\U0001f6cc\U0001f3fb|\U0001f6cc\U0001f3fc|\U0001f6cc\U0001f3fd|\U0001f6cc\U0001f3fe|\U0001f6cc\U0001f3ff|"
            r"\U0001f46d\U0001f3fb|\U0001f46d\U0001f3fc|\U0001f46d\U0001f3fd|\U0001f46d\U0001f3fe|\U0001f46d\U0001f3ff|"
            r"\U0001f46b\U0001f3fb|\U0001f46b\U0001f3fc|\U0001f46b\U0001f3fd|\U0001f46b\U0001f3fe|\U0001f46b\U0001f3ff|"
            r"\U0001f46c\U0001f3fb|\U0001f46c\U0001f3fc|\U0001f46c\U0001f3fd|\U0001f46c\U0001f3fe|\U0001f46c\U0001f3ff|"
            r"\U0001f5e3\ufe0f|\U0001f43f\ufe0f|\U0001f54a\ufe0f|\U0001f577\ufe0f|\U0001f578\ufe0f|\U0001f3f5\ufe0f|"
            r"\u2618\ufe0f|\U0001f336\ufe0f|\U0001f37d\ufe0f|\U0001f5fa\ufe0f|\U0001f3d4\ufe0f|\u26f0\ufe0f|\U0001f3d5"
            r"\ufe0f|\U0001f3d6\ufe0f|\U0001f3dc\ufe0f|\U0001f3dd\ufe0f|\U0001f3de\ufe0f|\U0001f3df\ufe0f|\U0001f3db"
            r"\ufe0f|\U0001f3d7\ufe0f|\U0001f3d8\ufe0f|\U0001f3da\ufe0f|\u26e9\ufe0f|\U0001f3d9\ufe0f|\u2668\ufe0f|"
            r"\U0001f3ce\ufe0f|\U0001f3cd\ufe0f|\U0001f6e3\ufe0f|\U0001f6e4\ufe0f|\U0001f6e2\ufe0f|\U0001f6f3\ufe0f|"
            r"\u26f4\ufe0f|\U0001f6e5\ufe0f|\u2708\ufe0f|\U0001f6e9\ufe0f|\U0001f6f0\ufe0f|\U0001f6ce\ufe0f|\u23f1"
            r"\ufe0f|\u23f2\ufe0f|\U0001f570\ufe0f|\U0001f321\ufe0f|\u2600\ufe0f|\u2601\ufe0f|\u26c8\ufe0f|\U0001f324"
            r"\ufe0f|\U0001f325\ufe0f|\U0001f326\ufe0f|\U0001f327\ufe0f|\U0001f328\ufe0f|\U0001f329\ufe0f|\U0001f32a"
            r"\ufe0f|\U0001f32b\ufe0f|\U0001f32c\ufe0f|\u2602\ufe0f|\u26f1\ufe0f|\u2744\ufe0f|\u2603\ufe0f|\u2604"
            r"\ufe0f|\U0001f397\ufe0f|\U0001f39f\ufe0f|\U0001f396\ufe0f|\u26f8\ufe0f|\U0001f579\ufe0f|\u2660\ufe0f|"
            r"\u2665\ufe0f|\u2666\ufe0f|\u2663\ufe0f|\u265f\ufe0f|\U0001f5bc\ufe0f|\U0001f576\ufe0f|\U0001f6cd\ufe0f|"
            r"\u26d1\ufe0f|\U0001f399\ufe0f|\U0001f39a\ufe0f|\U0001f39b\ufe0f|\u260e\ufe0f|\U0001f5a5\ufe0f|\U0001f5a8"
            r"\ufe0f|\u2328\ufe0f|\U0001f5b1\ufe0f|\U0001f5b2\ufe0f|\U0001f39e\ufe0f|\U0001f4fd\ufe0f|\U0001f56f\ufe0f|"
            r"\U0001f5de\ufe0f|\U0001f3f7\ufe0f|\u2709\ufe0f|\U0001f5f3\ufe0f|\u270f\ufe0f|\u2712\ufe0f|\U0001f58b"
            r"\ufe0f|\U0001f58a\ufe0f|\U0001f58c\ufe0f|\U0001f58d\ufe0f|\U0001f5c2\ufe0f|\U0001f5d2\ufe0f|\U0001f5d3"
            r"\ufe0f|\U0001f587\ufe0f|\u2702\ufe0f|\U0001f5c3\ufe0f|\U0001f5c4\ufe0f|\U0001f5d1\ufe0f|\U0001f5dd\ufe0f|"
            r"\u26cf\ufe0f|\u2692\ufe0f|\U0001f6e0\ufe0f|\U0001f5e1\ufe0f|\u2694\ufe0f|\U0001f6e1\ufe0f|\u2699\ufe0f|"
            r"\U0001f5dc\ufe0f|\u2696\ufe0f|\u26d3\ufe0f|\u2697\ufe0f|\U0001f6cf\ufe0f|\U0001f6cb\ufe0f|\u26b0\ufe0f|"
            r"\u26b1\ufe0f|\u26a0\ufe0f|\u2622\ufe0f|\u2623\ufe0f|\u2b06\ufe0f|\u2197\ufe0f|\u27a1\ufe0f|\u2198\ufe0f|"
            r"\u2b07\ufe0f|\u2199\ufe0f|\u2b05\ufe0f|\u2196\ufe0f|\u2195\ufe0f|\u2194\ufe0f|\u21a9\ufe0f|\u21aa\ufe0f|"
            r"\u2934\ufe0f|\u2935\ufe0f|\u269b\ufe0f|\U0001f549\ufe0f|\u2721\ufe0f|\u2638\ufe0f|\u262f\ufe0f|\u271d"
            r"\ufe0f|\u2626\ufe0f|\u262a\ufe0f|\u262e\ufe0f|\u25b6\ufe0f|\u23ed\ufe0f|\u23ef\ufe0f|\u25c0\ufe0f|\u23ee"
            r"\ufe0f|\u23f8\ufe0f|\u23f9\ufe0f|\u23fa\ufe0f|\u23cf\ufe0f|\u2640\ufe0f|\u2642\ufe0f|\u2695\ufe0f|\u267e"
            r"\ufe0f|\u267b\ufe0f|\u269c\ufe0f|\u2611\ufe0f|\u2714\ufe0f|\u2716\ufe0f|\u303d\ufe0f|\u2733\ufe0f|\u2734"
            r"\ufe0f|\u2747\ufe0f|\u203c\ufe0f|\u2049\ufe0f|\u3030\ufe0f|\xa9\ufe0f|\xae\ufe0f|\u2122\ufe0f|\\#\u20e3|"
            r"\\*\u20e3|0\u20e3|1\u20e3|2\u20e3|3\u20e3|4\u20e3|5\u20e3|6\u20e3|7\u20e3|8\u20e3|9\u20e3|\U0001f170"
            r"\ufe0f|\U0001f171\ufe0f|\u2139\ufe0f|\u24c2\ufe0f|\U0001f17e\ufe0f|\U0001f17f\ufe0f|\U0001f202\ufe0f|"
            r"\U0001f237\ufe0f|\u3297\ufe0f|\u3299\ufe0f|\u25fc\ufe0f|\u25fb\ufe0f|\u25aa\ufe0f|\u25ab\ufe0f|\U0001f3f3"
            r"\ufe0f|\U0001f1e6\U0001f1e8|\U0001f1e6\U0001f1e9|\U0001f1e6\U0001f1ea|\U0001f1e6\U0001f1eb|\U0001f1e6"
            r"\U0001f1ec|\U0001f1e6\U0001f1ee|\U0001f1e6\U0001f1f1|\U0001f1e6\U0001f1f2|\U0001f1e6\U0001f1f4|\U0001f1e6"
            r"\U0001f1f6|\U0001f1e6\U0001f1f7|\U0001f1e6\U0001f1f8|\U0001f1e6\U0001f1f9|\U0001f1e6\U0001f1fa|\U0001f1e6"
            r"\U0001f1fc|\U0001f1e6\U0001f1fd|\U0001f1e6\U0001f1ff|\U0001f1e7\U0001f1e6|\U0001f1e7\U0001f1e7|\U0001f1e7"
            r"\U0001f1e9|\U0001f1e7\U0001f1ea|\U0001f1e7\U0001f1eb|\U0001f1e7\U0001f1ec|\U0001f1e7\U0001f1ed|\U0001f1e7"
            r"\U0001f1ee|\U0001f1e7\U0001f1ef|\U0001f1e7\U0001f1f1|\U0001f1e7\U0001f1f2|\U0001f1e7\U0001f1f3|\U0001f1e7"
            r"\U0001f1f4|\U0001f1e7\U0001f1f6|\U0001f1e7\U0001f1f7|\U0001f1e7\U0001f1f8|\U0001f1e7\U0001f1f9|\U0001f1e7"
            r"\U0001f1fb|\U0001f1e7\U0001f1fc|\U0001f1e7\U0001f1fe|\U0001f1e7\U0001f1ff|\U0001f1e8\U0001f1e6|\U0001f1e8"
            r"\U0001f1e8|\U0001f1e8\U0001f1e9|\U0001f1e8\U0001f1eb|\U0001f1e8\U0001f1ec|\U0001f1e8\U0001f1ed|\U0001f1e8"
            r"\U0001f1ee|\U0001f1e8\U0001f1f0|\U0001f1e8\U0001f1f1|\U0001f1e8\U0001f1f2|\U0001f1e8\U0001f1f3|\U0001f1e8"
            r"\U0001f1f4|\U0001f1e8\U0001f1f5|\U0001f1e8\U0001f1f7|\U0001f1e8\U0001f1fa|\U0001f1e8\U0001f1fb|\U0001f1e8"
            r"\U0001f1fc|\U0001f1e8\U0001f1fd|\U0001f1e8\U0001f1fe|\U0001f1e8\U0001f1ff|\U0001f1e9\U0001f1ea|\U0001f1e9"
            r"\U0001f1ec|\U0001f1e9\U0001f1ef|\U0001f1e9\U0001f1f0|\U0001f1e9\U0001f1f2|\U0001f1e9\U0001f1f4|\U0001f1e9"
            r"\U0001f1ff|\U0001f1ea\U0001f1e6|\U0001f1ea\U0001f1e8|\U0001f1ea\U0001f1ea|\U0001f1ea\U0001f1ec|\U0001f1ea"
            r"\U0001f1ed|\U0001f1ea\U0001f1f7|\U0001f1ea\U0001f1f8|\U0001f1ea\U0001f1f9|\U0001f1ea\U0001f1fa|\U0001f1eb"
            r"\U0001f1ee|\U0001f1eb\U0001f1ef|\U0001f1eb\U0001f1f0|\U0001f1eb\U0001f1f2|\U0001f1eb\U0001f1f4|\U0001f1eb"
            r"\U0001f1f7|\U0001f1ec\U0001f1e6|\U0001f1ec\U0001f1e7|\U0001f1ec\U0001f1e9|\U0001f1ec\U0001f1ea|\U0001f1ec"
            r"\U0001f1eb|\U0001f1ec\U0001f1ec|\U0001f1ec\U0001f1ed|\U0001f1ec\U0001f1ee|\U0001f1ec\U0001f1f1|\U0001f1ec"
            r"\U0001f1f2|\U0001f1ec\U0001f1f3|\U0001f1ec\U0001f1f5|\U0001f1ec\U0001f1f6|\U0001f1ec\U0001f1f7|\U0001f1ec"
            r"\U0001f1f8|\U0001f1ec\U0001f1f9|\U0001f1ec\U0001f1fa|\U0001f1ec\U0001f1fc|\U0001f1ec\U0001f1fe|\U0001f1ed"
            r"\U0001f1f0|\U0001f1ed\U0001f1f2|\U0001f1ed\U0001f1f3|\U0001f1ed\U0001f1f7|\U0001f1ed\U0001f1f9|\U0001f1ed"
            r"\U0001f1fa|\U0001f1ee\U0001f1e8|\U0001f1ee\U0001f1e9|\U0001f1ee\U0001f1ea|\U0001f1ee\U0001f1f1|\U0001f1ee"
            r"\U0001f1f2|\U0001f1ee\U0001f1f3|\U0001f1ee\U0001f1f4|\U0001f1ee\U0001f1f6|\U0001f1ee\U0001f1f7|\U0001f1ee"
            r"\U0001f1f8|\U0001f1ee\U0001f1f9|\U0001f1ef\U0001f1ea|\U0001f1ef\U0001f1f2|\U0001f1ef\U0001f1f4|\U0001f1ef"
            r"\U0001f1f5|\U0001f1f0\U0001f1ea|\U0001f1f0\U0001f1ec|\U0001f1f0\U0001f1ed|\U0001f1f0\U0001f1ee|\U0001f1f0"
            r"\U0001f1f2|\U0001f1f0\U0001f1f3|\U0001f1f0\U0001f1f5|\U0001f1f0\U0001f1f7|\U0001f1f0\U0001f1fc|\U0001f1f0"
            r"\U0001f1fe|\U0001f1f0\U0001f1ff|\U0001f1f1\U0001f1e6|\U0001f1f1\U0001f1e7|\U0001f1f1\U0001f1e8|\U0001f1f1"
            r"\U0001f1ee|\U0001f1f1\U0001f1f0|\U0001f1f1\U0001f1f7|\U0001f1f1\U0001f1f8|\U0001f1f1\U0001f1f9|\U0001f1f1"
            r"\U0001f1fa|\U0001f1f1\U0001f1fb|\U0001f1f1\U0001f1fe|\U0001f1f2\U0001f1e6|\U0001f1f2\U0001f1e8|\U0001f1f2"
            r"\U0001f1e9|\U0001f1f2\U0001f1ea|\U0001f1f2\U0001f1eb|\U0001f1f2\U0001f1ec|\U0001f1f2\U0001f1ed|\U0001f1f2"
            r"\U0001f1f0|\U0001f1f2\U0001f1f1|\U0001f1f2\U0001f1f2|\U0001f1f2\U0001f1f3|\U0001f1f2\U0001f1f4|\U0001f1f2"
            r"\U0001f1f5|\U0001f1f2\U0001f1f6|\U0001f1f2\U0001f1f7|\U0001f1f2\U0001f1f8|\U0001f1f2\U0001f1f9|\U0001f1f2"
            r"\U0001f1fa|\U0001f1f2\U0001f1fb|\U0001f1f2\U0001f1fc|\U0001f1f2\U0001f1fd|\U0001f1f2\U0001f1fe|\U0001f1f2"
            r"\U0001f1ff|\U0001f1f3\U0001f1e6|\U0001f1f3\U0001f1e8|\U0001f1f3\U0001f1ea|\U0001f1f3\U0001f1eb|\U0001f1f3"
            r"\U0001f1ec|\U0001f1f3\U0001f1ee|\U0001f1f3\U0001f1f1|\U0001f1f3\U0001f1f4|\U0001f1f3\U0001f1f5|\U0001f1f3"
            r"\U0001f1f7|\U0001f1f3\U0001f1fa|\U0001f1f3\U0001f1ff|\U0001f1f4\U0001f1f2|\U0001f1f5\U0001f1e6|\U0001f1f5"
            r"\U0001f1ea|\U0001f1f5\U0001f1eb|\U0001f1f5\U0001f1ec|\U0001f1f5\U0001f1ed|\U0001f1f5\U0001f1f0|\U0001f1f5"
            r"\U0001f1f1|\U0001f1f5\U0001f1f2|\U0001f1f5\U0001f1f3|\U0001f1f5\U0001f1f7|\U0001f1f5\U0001f1f8|\U0001f1f5"
            r"\U0001f1f9|\U0001f1f5\U0001f1fc|\U0001f1f5\U0001f1fe|\U0001f1f6\U0001f1e6|\U0001f1f7\U0001f1ea|\U0001f1f7"
            r"\U0001f1f4|\U0001f1f7\U0001f1f8|\U0001f1f7\U0001f1fa|\U0001f1f7\U0001f1fc|\U0001f1f8\U0001f1e6|\U0001f1f8"
            r"\U0001f1e7|\U0001f1f8\U0001f1e8|\U0001f1f8\U0001f1e9|\U0001f1f8\U0001f1ea|\U0001f1f8\U0001f1ec|\U0001f1f8"
            r"\U0001f1ed|\U0001f1f8\U0001f1ee|\U0001f1f8\U0001f1ef|\U0001f1f8\U0001f1f0|\U0001f1f8\U0001f1f1|\U0001f1f8"
            r"\U0001f1f2|\U0001f1f8\U0001f1f3|\U0001f1f8\U0001f1f4|\U0001f1f8\U0001f1f7|\U0001f1f8\U0001f1f8|\U0001f1f8"
            r"\U0001f1f9|\U0001f1f8\U0001f1fb|\U0001f1f8\U0001f1fd|\U0001f1f8\U0001f1fe|\U0001f1f8\U0001f1ff|\U0001f1f9"
            r"\U0001f1e6|\U0001f1f9\U0001f1e8|\U0001f1f9\U0001f1e9|\U0001f1f9\U0001f1eb|\U0001f1f9\U0001f1ec|\U0001f1f9"
            r"\U0001f1ed|\U0001f1f9\U0001f1ef|\U0001f1f9\U0001f1f0|\U0001f1f9\U0001f1f1|\U0001f1f9\U0001f1f2|\U0001f1f9"
            r"\U0001f1f3|\U0001f1f9\U0001f1f4|\U0001f1f9\U0001f1f7|\U0001f1f9\U0001f1f9|\U0001f1f9\U0001f1fb|\U0001f1f9"
            r"\U0001f1fc|\U0001f1f9\U0001f1ff|\U0001f1fa\U0001f1e6|\U0001f1fa\U0001f1ec|\U0001f1fa\U0001f1f2|\U0001f1fa"
            r"\U0001f1f3|\U0001f1fa\U0001f1f8|\U0001f1fa\U0001f1fe|\U0001f1fa\U0001f1ff|\U0001f1fb\U0001f1e6|\U0001f1fb"
            r"\U0001f1e8|\U0001f1fb\U0001f1ea|\U0001f1fb\U0001f1ec|\U0001f1fb\U0001f1ee|\U0001f1fb\U0001f1f3|\U0001f1fb"
            r"\U0001f1fa|\U0001f1fc\U0001f1eb|\U0001f1fc\U0001f1f8|\U0001f1fd\U0001f1f0|\U0001f1fe\U0001f1ea|\U0001f1fe"
            r"\U0001f1f9|\U0001f1ff\U0001f1e6|\U0001f1ff\U0001f1f2|\U0001f1ff\U0001f1fc|\U0001f600|\U0001f603|"
            r"\U0001f604|\U0001f601|\U0001f606|\U0001f605|\U0001f923|\U0001f602|\U0001f642|\U0001f643|\U0001f609|"
            r"\U0001f60a|\U0001f607|\U0001f970|\U0001f60d|\U0001f929|\U0001f618|\U0001f617|\u263a|\U0001f61a|"
            r"\U0001f619|\U0001f60b|\U0001f61b|\U0001f61c|\U0001f92a|\U0001f61d|\U0001f911|\U0001f917|\U0001f92d|"
            r"\U0001f92b|\U0001f914|\U0001f910|\U0001f928|\U0001f610|\U0001f611|\U0001f636|\U0001f60f|\U0001f612|"
            r"\U0001f644|\U0001f62c|\U0001f925|\U0001f60c|\U0001f614|\U0001f62a|\U0001f924|\U0001f634|\U0001f637|"
            r"\U0001f912|\U0001f915|\U0001f922|\U0001f92e|\U0001f927|\U0001f975|\U0001f976|\U0001f974|\U0001f635|"
            r"\U0001f92f|\U0001f920|\U0001f973|\U0001f60e|\U0001f913|\U0001f9d0|\U0001f615|\U0001f61f|\U0001f641|"
            r"\u2639|\U0001f62e|\U0001f62f|\U0001f632|\U0001f633|\U0001f97a|\U0001f626|\U0001f627|\U0001f628|"
            r"\U0001f630|\U0001f625|\U0001f622|\U0001f62d|\U0001f631|\U0001f616|\U0001f623|\U0001f61e|\U0001f613|"
            r"\U0001f629|\U0001f62b|\U0001f971|\U0001f624|\U0001f621|\U0001f620|\U0001f92c|\U0001f608|\U0001f47f|"
            r"\U0001f480|\u2620|\U0001f4a9|\U0001f921|\U0001f479|\U0001f47a|\U0001f47b|\U0001f47d|\U0001f47e|"
            r"\U0001f916|\U0001f63a|\U0001f638|\U0001f639|\U0001f63b|\U0001f63c|\U0001f63d|\U0001f640|\U0001f63f|"
            r"\U0001f63e|\U0001f648|\U0001f649|\U0001f64a|\U0001f48b|\U0001f48c|\U0001f498|\U0001f49d|\U0001f496|"
            r"\U0001f497|\U0001f493|\U0001f49e|\U0001f495|\U0001f49f|\u2763|\U0001f494|\u2764|\U0001f9e1|\U0001f49b|"
            r"\U0001f49a|\U0001f499|\U0001f49c|\U0001f90e|\U0001f5a4|\U0001f90d|\U0001f4af|\U0001f4a2|\U0001f4a5|"
            r"\U0001f4ab|\U0001f4a6|\U0001f4a8|\U0001f573|\U0001f4a3|\U0001f4ac|\U0001f5e8|\U0001f5ef|\U0001f4ad|"
            r"\U0001f4a4|\U0001f44b|\U0001f91a|\U0001f590|\u270b|\U0001f596|\U0001f44c|\U0001f90f|\u270c|\U0001f91e|"
            r"\U0001f91f|\U0001f918|\U0001f919|\U0001f448|\U0001f449|\U0001f446|\U0001f595|\U0001f447|\u261d|"
            r"\U0001f44d|\U0001f44e|\u270a|\U0001f44a|\U0001f91b|\U0001f91c|\U0001f44f|\U0001f64c|\U0001f450|"
            r"\U0001f932|\U0001f91d|\U0001f64f|\u270d|\U0001f485|\U0001f933|\U0001f4aa|\U0001f9be|\U0001f9bf|"
            r"\U0001f9b5|\U0001f9b6|\U0001f442|\U0001f9bb|\U0001f443|\U0001f9e0|\U0001f9b7|\U0001f9b4|\U0001f440|"
            r"\U0001f441|\U0001f445|\U0001f444|\U0001f476|\U0001f9d2|\U0001f466|\U0001f467|\U0001f9d1|\U0001f471|"
            r"\U0001f468|\U0001f9d4|\U0001f469|\U0001f9d3|\U0001f474|\U0001f475|\U0001f64d|\U0001f64e|\U0001f645|"
            r"\U0001f646|\U0001f481|\U0001f64b|\U0001f9cf|\U0001f647|\U0001f926|\U0001f937|\U0001f46e|\U0001f575|"
            r"\U0001f482|\U0001f477|\U0001f934|\U0001f478|\U0001f473|\U0001f472|\U0001f9d5|\U0001f935|\U0001f470|"
            r"\U0001f930|\U0001f931|\U0001f47c|\U0001f385|\U0001f936|\U0001f9b8|\U0001f9b9|\U0001f9d9|\U0001f9da|"
            r"\U0001f9db|\U0001f9dc|\U0001f9dd|\U0001f9de|\U0001f9df|\U0001f486|\U0001f487|\U0001f6b6|\U0001f9cd|"
            r"\U0001f9ce|\U0001f3c3|\U0001f483|\U0001f57a|\U0001f574|\U0001f46f|\U0001f9d6|\U0001f9d7|\U0001f93a|"
            r"\U0001f3c7|\u26f7|\U0001f3c2|\U0001f3cc|\U0001f3c4|\U0001f6a3|\U0001f3ca|\u26f9|\U0001f3cb|\U0001f6b4|"
            r"\U0001f6b5|\U0001f938|\U0001f93c|\U0001f93d|\U0001f93e|\U0001f939|\U0001f9d8|\U0001f6c0|\U0001f6cc|"
            r"\U0001f46d|\U0001f46b|\U0001f46c|\U0001f48f|\U0001f491|\U0001f46a|\U0001f5e3|\U0001f464|\U0001f465|"
            r"\U0001f463|\U0001f3fb|\U0001f3fc|\U0001f3fd|\U0001f3fe|\U0001f3ff|\U0001f9b0|\U0001f9b1|\U0001f9b3|"
            r"\U0001f9b2|\U0001f435|\U0001f412|\U0001f98d|\U0001f9a7|\U0001f436|\U0001f415|\U0001f9ae|\U0001f429|"
            r"\U0001f43a|\U0001f98a|\U0001f99d|\U0001f431|\U0001f408|\U0001f981|\U0001f42f|\U0001f405|\U0001f406|"
            r"\U0001f434|\U0001f40e|\U0001f984|\U0001f993|\U0001f98c|\U0001f42e|\U0001f402|\U0001f403|\U0001f404|"
            r"\U0001f437|\U0001f416|\U0001f417|\U0001f43d|\U0001f40f|\U0001f411|\U0001f410|\U0001f42a|\U0001f42b|"
            r"\U0001f999|\U0001f992|\U0001f418|\U0001f98f|\U0001f99b|\U0001f42d|\U0001f401|\U0001f400|\U0001f439|"
            r"\U0001f430|\U0001f407|\U0001f43f|\U0001f994|\U0001f987|\U0001f43b|\U0001f428|\U0001f43c|\U0001f9a5|"
            r"\U0001f9a6|\U0001f9a8|\U0001f998|\U0001f9a1|\U0001f43e|\U0001f983|\U0001f414|\U0001f413|\U0001f423|"
            r"\U0001f424|\U0001f425|\U0001f426|\U0001f427|\U0001f54a|\U0001f985|\U0001f986|\U0001f9a2|\U0001f989|"
            r"\U0001f9a9|\U0001f99a|\U0001f99c|\U0001f438|\U0001f40a|\U0001f422|\U0001f98e|\U0001f40d|\U0001f432|"
            r"\U0001f409|\U0001f995|\U0001f996|\U0001f433|\U0001f40b|\U0001f42c|\U0001f41f|\U0001f420|\U0001f421|"
            r"\U0001f988|\U0001f419|\U0001f41a|\U0001f40c|\U0001f98b|\U0001f41b|\U0001f41c|\U0001f41d|\U0001f41e|"
            r"\U0001f997|\U0001f577|\U0001f578|\U0001f982|\U0001f99f|\U0001f9a0|\U0001f490|\U0001f338|\U0001f4ae|"
            r"\U0001f3f5|\U0001f339|\U0001f940|\U0001f33a|\U0001f33b|\U0001f33c|\U0001f337|\U0001f331|\U0001f332|"
            r"\U0001f333|\U0001f334|\U0001f335|\U0001f33e|\U0001f33f|\u2618|\U0001f340|\U0001f341|\U0001f342|"
            r"\U0001f343|\U0001f347|\U0001f348|\U0001f349|\U0001f34a|\U0001f34b|\U0001f34c|\U0001f34d|\U0001f96d|"
            r"\U0001f34e|\U0001f34f|\U0001f350|\U0001f351|\U0001f352|\U0001f353|\U0001f95d|\U0001f345|\U0001f965|"
            r"\U0001f951|\U0001f346|\U0001f954|\U0001f955|\U0001f33d|\U0001f336|\U0001f952|\U0001f96c|\U0001f966|"
            r"\U0001f9c4|\U0001f9c5|\U0001f344|\U0001f95c|\U0001f330|\U0001f35e|\U0001f950|\U0001f956|\U0001f968|"
            r"\U0001f96f|\U0001f95e|\U0001f9c7|\U0001f9c0|\U0001f356|\U0001f357|\U0001f969|\U0001f953|\U0001f354|"
            r"\U0001f35f|\U0001f355|\U0001f32d|\U0001f96a|\U0001f32e|\U0001f32f|\U0001f959|\U0001f9c6|\U0001f95a|"
            r"\U0001f373|\U0001f958|\U0001f372|\U0001f963|\U0001f957|\U0001f37f|\U0001f9c8|\U0001f9c2|\U0001f96b|"
            r"\U0001f371|\U0001f358|\U0001f359|\U0001f35a|\U0001f35b|\U0001f35c|\U0001f35d|\U0001f360|\U0001f362|"
            r"\U0001f363|\U0001f364|\U0001f365|\U0001f96e|\U0001f361|\U0001f95f|\U0001f960|\U0001f961|\U0001f980|"
            r"\U0001f99e|\U0001f990|\U0001f991|\U0001f9aa|\U0001f366|\U0001f367|\U0001f368|\U0001f369|\U0001f36a|"
            r"\U0001f382|\U0001f370|\U0001f9c1|\U0001f967|\U0001f36b|\U0001f36c|\U0001f36d|\U0001f36e|\U0001f36f|"
            r"\U0001f37c|\U0001f95b|\u2615|\U0001f375|\U0001f376|\U0001f37e|\U0001f377|\U0001f378|\U0001f379|"
            r"\U0001f37a|\U0001f37b|\U0001f942|\U0001f943|\U0001f964|\U0001f9c3|\U0001f9c9|\U0001f9ca|\U0001f962|"
            r"\U0001f37d|\U0001f374|\U0001f944|\U0001f52a|\U0001f3fa|\U0001f30d|\U0001f30e|\U0001f30f|\U0001f310|"
            r"\U0001f5fa|\U0001f5fe|\U0001f9ed|\U0001f3d4|\u26f0|\U0001f30b|\U0001f5fb|\U0001f3d5|\U0001f3d6|"
            r"\U0001f3dc|\U0001f3dd|\U0001f3de|\U0001f3df|\U0001f3db|\U0001f3d7|\U0001f9f1|\U0001f3d8|\U0001f3da|"
            r"\U0001f3e0|\U0001f3e1|\U0001f3e2|\U0001f3e3|\U0001f3e4|\U0001f3e5|\U0001f3e6|\U0001f3e8|\U0001f3e9|"
            r"\U0001f3ea|\U0001f3eb|\U0001f3ec|\U0001f3ed|\U0001f3ef|\U0001f3f0|\U0001f492|\U0001f5fc|\U0001f5fd|"
            r"\u26ea|\U0001f54c|\U0001f6d5|\U0001f54d|\u26e9|\U0001f54b|\u26f2|\u26fa|\U0001f301|\U0001f303|\U0001f3d9|"
            r"\U0001f304|\U0001f305|\U0001f306|\U0001f307|\U0001f309|\u2668|\U0001f3a0|\U0001f3a1|\U0001f3a2|"
            r"\U0001f488|\U0001f3aa|\U0001f682|\U0001f683|\U0001f684|\U0001f685|\U0001f686|\U0001f687|\U0001f688|"
            r"\U0001f689|\U0001f68a|\U0001f69d|\U0001f69e|\U0001f68b|\U0001f68c|\U0001f68d|\U0001f68e|\U0001f690|"
            r"\U0001f691|\U0001f692|\U0001f693|\U0001f694|\U0001f695|\U0001f696|\U0001f697|\U0001f698|\U0001f699|"
            r"\U0001f69a|\U0001f69b|\U0001f69c|\U0001f3ce|\U0001f3cd|\U0001f6f5|\U0001f9bd|\U0001f9bc|\U0001f6fa|"
            r"\U0001f6b2|\U0001f6f4|\U0001f6f9|\U0001f68f|\U0001f6e3|\U0001f6e4|\U0001f6e2|\u26fd|\U0001f6a8|"
            r"\U0001f6a5|\U0001f6a6|\U0001f6d1|\U0001f6a7|\u2693|\u26f5|\U0001f6f6|\U0001f6a4|\U0001f6f3|\u26f4|"
            r"\U0001f6e5|\U0001f6a2|\u2708|\U0001f6e9|\U0001f6eb|\U0001f6ec|\U0001fa82|\U0001f4ba|\U0001f681|"
            r"\U0001f69f|\U0001f6a0|\U0001f6a1|\U0001f6f0|\U0001f680|\U0001f6f8|\U0001f6ce|\U0001f9f3|\u231b|\u23f3|"
            r"\u231a|\u23f0|\u23f1|\u23f2|\U0001f570|\U0001f55b|\U0001f567|\U0001f550|\U0001f55c|\U0001f551|\U0001f55d|"
            r"\U0001f552|\U0001f55e|\U0001f553|\U0001f55f|\U0001f554|\U0001f560|\U0001f555|\U0001f561|\U0001f556|"
            r"\U0001f562|\U0001f557|\U0001f563|\U0001f558|\U0001f564|\U0001f559|\U0001f565|\U0001f55a|\U0001f566|"
            r"\U0001f311|\U0001f312|\U0001f313|\U0001f314|\U0001f315|\U0001f316|\U0001f317|\U0001f318|\U0001f319|"
            r"\U0001f31a|\U0001f31b|\U0001f31c|\U0001f321|\u2600|\U0001f31d|\U0001f31e|\U0001fa90|\u2b50|\U0001f31f|"
            r"\U0001f320|\U0001f30c|\u2601|\u26c5|\u26c8|\U0001f324|\U0001f325|\U0001f326|\U0001f327|\U0001f328|"
            r"\U0001f329|\U0001f32a|\U0001f32b|\U0001f32c|\U0001f300|\U0001f308|\U0001f302|\u2602|\u2614|\u26f1|"
            r"\u26a1|\u2744|\u2603|\u26c4|\u2604|\U0001f525|\U0001f4a7|\U0001f30a|\U0001f383|\U0001f384|\U0001f386|"
            r"\U0001f387|\U0001f9e8|\u2728|\U0001f388|\U0001f389|\U0001f38a|\U0001f38b|\U0001f38d|\U0001f38e|"
            r"\U0001f38f|\U0001f390|\U0001f391|\U0001f9e7|\U0001f380|\U0001f381|\U0001f397|\U0001f39f|\U0001f3ab|"
            r"\U0001f396|\U0001f3c6|\U0001f3c5|\U0001f947|\U0001f948|\U0001f949|\u26bd|\u26be|\U0001f94e|\U0001f3c0|"
            r"\U0001f3d0|\U0001f3c8|\U0001f3c9|\U0001f3be|\U0001f94f|\U0001f3b3|\U0001f3cf|\U0001f3d1|\U0001f3d2|"
            r"\U0001f94d|\U0001f3d3|\U0001f3f8|\U0001f94a|\U0001f94b|\U0001f945|\u26f3|\u26f8|\U0001f3a3|\U0001f93f|"
            r"\U0001f3bd|\U0001f3bf|\U0001f6f7|\U0001f94c|\U0001f3af|\U0001fa80|\U0001fa81|\U0001f3b1|\U0001f52e|"
            r"\U0001f9ff|\U0001f3ae|\U0001f579|\U0001f3b0|\U0001f3b2|\U0001f9e9|\U0001f9f8|\u2660|\u2665|\u2666|\u2663|"
            r"\u265f|\U0001f0cf|\U0001f004|\U0001f3b4|\U0001f3ad|\U0001f5bc|\U0001f3a8|\U0001f9f5|\U0001f9f6|"
            r"\U0001f453|\U0001f576|\U0001f97d|\U0001f97c|\U0001f9ba|\U0001f454|\U0001f455|\U0001f456|\U0001f9e3|"
            r"\U0001f9e4|\U0001f9e5|\U0001f9e6|\U0001f457|\U0001f458|\U0001f97b|\U0001fa71|\U0001fa72|\U0001fa73|"
            r"\U0001f459|\U0001f45a|\U0001f45b|\U0001f45c|\U0001f45d|\U0001f6cd|\U0001f392|\U0001f45e|\U0001f45f|"
            r"\U0001f97e|\U0001f97f|\U0001f460|\U0001f461|\U0001fa70|\U0001f462|\U0001f451|\U0001f452|\U0001f3a9|"
            r"\U0001f393|\U0001f9e2|\u26d1|\U0001f4ff|\U0001f484|\U0001f48d|\U0001f48e|\U0001f507|\U0001f508|"
            r"\U0001f509|\U0001f50a|\U0001f4e2|\U0001f4e3|\U0001f4ef|\U0001f514|\U0001f515|\U0001f3bc|\U0001f3b5|"
            r"\U0001f3b6|\U0001f399|\U0001f39a|\U0001f39b|\U0001f3a4|\U0001f3a7|\U0001f4fb|\U0001f3b7|\U0001f3b8|"
            r"\U0001f3b9|\U0001f3ba|\U0001f3bb|\U0001fa95|\U0001f941|\U0001f4f1|\U0001f4f2|\u260e|\U0001f4de|"
            r"\U0001f4df|\U0001f4e0|\U0001f50b|\U0001f50c|\U0001f4bb|\U0001f5a5|\U0001f5a8|\u2328|\U0001f5b1|"
            r"\U0001f5b2|\U0001f4bd|\U0001f4be|\U0001f4bf|\U0001f4c0|\U0001f9ee|\U0001f3a5|\U0001f39e|\U0001f4fd|"
            r"\U0001f3ac|\U0001f4fa|\U0001f4f7|\U0001f4f8|\U0001f4f9|\U0001f4fc|\U0001f50d|\U0001f50e|\U0001f56f|"
            r"\U0001f4a1|\U0001f526|\U0001f3ee|\U0001fa94|\U0001f4d4|\U0001f4d5|\U0001f4d6|\U0001f4d7|\U0001f4d8|"
            r"\U0001f4d9|\U0001f4da|\U0001f4d3|\U0001f4d2|\U0001f4c3|\U0001f4dc|\U0001f4c4|\U0001f4f0|\U0001f5de|"
            r"\U0001f4d1|\U0001f516|\U0001f3f7|\U0001f4b0|\U0001f4b4|\U0001f4b5|\U0001f4b6|\U0001f4b7|\U0001f4b8|"
            r"\U0001f4b3|\U0001f9fe|\U0001f4b9|\U0001f4b1|\U0001f4b2|\u2709|\U0001f4e7|\U0001f4e8|\U0001f4e9|"
            r"\U0001f4e4|\U0001f4e5|\U0001f4e6|\U0001f4eb|\U0001f4ea|\U0001f4ec|\U0001f4ed|\U0001f4ee|\U0001f5f3|"
            r"\u270f|\u2712|\U0001f58b|\U0001f58a|\U0001f58c|\U0001f58d|\U0001f4dd|\U0001f4bc|\U0001f4c1|\U0001f4c2|"
            r"\U0001f5c2|\U0001f4c5|\U0001f4c6|\U0001f5d2|\U0001f5d3|\U0001f4c7|\U0001f4c8|\U0001f4c9|\U0001f4ca|"
            r"\U0001f4cb|\U0001f4cc|\U0001f4cd|\U0001f4ce|\U0001f587|\U0001f4cf|\U0001f4d0|\u2702|\U0001f5c3|"
            r"\U0001f5c4|\U0001f5d1|\U0001f512|\U0001f513|\U0001f50f|\U0001f510|\U0001f511|\U0001f5dd|\U0001f528|"
            r"\U0001fa93|\u26cf|\u2692|\U0001f6e0|\U0001f5e1|\u2694|\U0001f52b|\U0001f3f9|\U0001f6e1|\U0001f527|"
            r"\U0001f529|\u2699|\U0001f5dc|\u2696|\U0001f9af|\U0001f517|\u26d3|\U0001f9f0|\U0001f9f2|\u2697|\U0001f9ea|"
            r"\U0001f9eb|\U0001f9ec|\U0001f52c|\U0001f52d|\U0001f4e1|\U0001f489|\U0001fa78|\U0001f48a|\U0001fa79|"
            r"\U0001fa7a|\U0001f6aa|\U0001f6cf|\U0001f6cb|\U0001fa91|\U0001f6bd|\U0001f6bf|\U0001f6c1|\U0001fa92|"
            r"\U0001f9f4|\U0001f9f7|\U0001f9f9|\U0001f9fa|\U0001f9fb|\U0001f9fc|\U0001f9fd|\U0001f9ef|\U0001f6d2|"
            r"\U0001f6ac|\u26b0|\u26b1|\U0001f5ff|\U0001f3e7|\U0001f6ae|\U0001f6b0|\u267f|\U0001f6b9|\U0001f6ba|"
            r"\U0001f6bb|\U0001f6bc|\U0001f6be|\U0001f6c2|\U0001f6c3|\U0001f6c4|\U0001f6c5|\u26a0|\U0001f6b8|\u26d4|"
            r"\U0001f6ab|\U0001f6b3|\U0001f6ad|\U0001f6af|\U0001f6b1|\U0001f6b7|\U0001f4f5|\U0001f51e|\u2622|\u2623|"
            r"\u2b06|\u2197|\u27a1|\u2198|\u2b07|\u2199|\u2b05|\u2196|\u2195|\u2194|\u21a9|\u21aa|\u2934|\u2935|"
            r"\U0001f503|\U0001f504|\U0001f519|\U0001f51a|\U0001f51b|\U0001f51c|\U0001f51d|\U0001f6d0|\u269b|"
            r"\U0001f549|\u2721|\u2638|\u262f|\u271d|\u2626|\u262a|\u262e|\U0001f54e|\U0001f52f|\u2648|\u2649|\u264a|"
            r"\u264b|\u264c|\u264d|\u264e|\u264f|\u2650|\u2651|\u2652|\u2653|\u26ce|\U0001f500|\U0001f501|\U0001f502|"
            r"\u25b6|\u23e9|\u23ed|\u23ef|\u25c0|\u23ea|\u23ee|\U0001f53c|\u23eb|\U0001f53d|\u23ec|\u23f8|\u23f9|"
            r"\u23fa|\u23cf|\U0001f3a6|\U0001f505|\U0001f506|\U0001f4f6|\U0001f4f3|\U0001f4f4|\u2640|\u2642|\u2695|"
            r"\u267e|\u267b|\u269c|\U0001f531|\U0001f4db|\U0001f530|\u2b55|\u2705|\u2611|\u2714|\u2716|\u274c|\u274e|"
            r"\u2795|\u2796|\u2797|\u27b0|\u27bf|\u303d|\u2733|\u2734|\u2747|\u203c|\u2049|\u2753|\u2754|\u2755|\u2757|"
            r"\u3030|\xa9|\xae|\u2122|\U0001f51f|\U0001f520|\U0001f521|\U0001f522|\U0001f523|\U0001f524|\U0001f170|"
            r"\U0001f18e|\U0001f171|\U0001f191|\U0001f192|\U0001f193|\u2139|\U0001f194|\u24c2|\U0001f195|\U0001f196|"
            r"\U0001f17e|\U0001f197|\U0001f17f|\U0001f198|\U0001f199|\U0001f19a|\U0001f201|\U0001f202|\U0001f237|"
            r"\U0001f236|\U0001f22f|\U0001f250|\U0001f239|\U0001f21a|\U0001f232|\U0001f251|\U0001f238|\U0001f234|"
            r"\U0001f233|\u3297|\u3299|\U0001f23a|\U0001f235|\U0001f534|\U0001f7e0|\U0001f7e1|\U0001f7e2|\U0001f535|"
            r"\U0001f7e3|\U0001f7e4|\u26ab|\u26aa|\U0001f7e5|\U0001f7e7|\U0001f7e8|\U0001f7e9|\U0001f7e6|\U0001f7ea|"
            r"\U0001f7eb|\u2b1b|\u2b1c|\u25fc|\u25fb|\u25fe|\u25fd|\u25aa|\u25ab|\U0001f536|\U0001f537|\U0001f538|"
            r"\U0001f539|\U0001f53a|\U0001f53b|\U0001f4a0|\U0001f518|\U0001f533|\U0001f532|\U0001f3c1|\U0001f6a9|"
            r"\U0001f38c|\U0001f3f4|\U0001f3f3")
        return emoji_pattern.sub(r' ', s)