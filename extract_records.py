"""Extract component records from the inspection report text files."""

import re
import csv
import os
from typing import Dict, List, Optional

TARGET_COMPONENTS = {'Deck', 'Superstructure', 'Substructure'}


class BridgeInspectionExtractor:
    def __init__(self):
        # Components to extract
        self.components = {
            '58': 'Deck',
            '59': 'Superstructure',
            '60': 'Substructure',
            '61': 'Channel Condition',
            '62': 'Culverts'
        }
        
        # Alternative component names used in different report formats
        self.component_aliases = {
            '61': ['Channel Condition', 'Channel/Channel Protection', 'Channel & Channel Protection']
        }

    def extract_structure_id(self, content: str) -> str:
        """Extract the structure ID from the report text."""
        match = re.search(r'Structure #(\d+)', content)
        return match.group(1) if match else "Unknown"

    def extract_component_rating(self, text: str, component_num: str) -> Optional[str]:
        """Extract the component rating."""
        # Format: 58 - Deck (8 - VERY GOOD CONDITION...)
        pattern = rf'{component_num}\s*-\s*[^(]*\((\d+)\s*-'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def is_page_header(self, line: str) -> bool:
        """Check whether a line is a page header."""
        header_patterns = [
            r'^===\s*Page\s+\d+\s*===$',  # === Page X ===
            r'^Route\s+\w+\s*/$',  # Route XXXX /
            r'^Structure\s+#\d+\s*/\s*\([^)]*\)$',  # Structure #XXXXX / (Routine)
            r'^[A-Z0-9]+\s+over\s+',  # XXXX over RIVER
            r'^Team Lead:',  # Team Lead: ...
            r'^.*Inspection Date:\s*\d+/\d+/\d+',  # Inspection Date
            r'^Town:\s*\d+\s*-\s*',  # Town: XXX - XXXXX
            r'^District\s+\d+',  # District info
            r'^Owner:\s*\d+\s*-\s*',  # Owner: X - ...
            r'^Maintenance Responsibility:\s*\d+\s*-\s*',  # Maintenance
            r'^Location:\s*',  # Location: ...
            r'^-?\d+\.\d+,\s*-?\d+\.\d+$',  # Coordinates
            r'OpenStreetMap'  # OpenStreetMap info
        ]

        return any(re.match(pattern, line, re.IGNORECASE) for pattern in header_patterns)

    def is_header_info(self, line: str) -> bool:
        """Check whether a line is report header information."""
        header_patterns = [
            r'^Route\s+[A-Z0-9\s]+/$',  # Route ALT VT100 /
            r'^Structure\s+#[A-Z0-9]+\s*/\s*\([^)]*\)$',  # Structure #0215A / (Routine)
            r'^[A-Z0-9]{2,}\s+[A-Z0-9]{2,}\s+[A-Z0-9\s]*\s+over\s+[A-Z]{2,}\s+[A-Z\s]+$',  # ALT VT 00100 ML over LAMOILLE RIVER (strict bridge name format)
            r'^Team Lead:',  # Team Lead: ...
            r'^.*Inspection Date:\s*\d+/\d+/\d+',  # Inspection Date
            r'^Town:\s*\d+\s*-\s*',  # Town: XXX - XXXXX
            r'^District\s+\d+',  # District info
            r'^Owner:\s*\d+\s*-\s*',  # Owner: X - ...
            r'^Maintenance Responsibility:\s*\d+\s*-\s*',  # Maintenance
            r'^Location:\s*',  # Location: ...
            r'^-?\d+\.\d+,\s*-?\d+\.\d+$',  # Coordinates
            r'OpenStreetMap',  # OpenStreetMap info
            r'^===\s*Page\s+\d+\s*===$',  # === Page X ===
            r'^(CHANNEL|SUBSTRUCTURE|DECK|SUPERSTRUCTURE|CULVERT|APPROACH|GENERAL|OBSERVATION)$'  # Section titles
        ]

        # Strict check for the bridge name format
        if re.match(r'^[A-Z0-9]{2,}\s+[A-Z0-9]{2,}\s+[A-Z0-9\s]*\s+over\s+[A-Z]{2,}\s+[A-Z\s]+$', line, re.IGNORECASE):
            # Make sure the line is not ordinary descriptive text
            # Lines with common descriptive words are not bridge names
            descriptive_words = ['loss', 'protection', 'erosion', 'damage', 'condition', 'minor', 'major', 'moderate']
            line_lower = line.lower()
            if any(word in line_lower for word in descriptive_words):
                return False
            return True

        return any(re.match(pattern, line, re.IGNORECASE) for pattern in header_patterns)

    def add_period_to_line(self, line: str, next_line: str = None) -> str:
        """Add a full stop to a line if it is missing."""
        if not line or not line.strip():
            return line
        
        line = line.strip()
        
        # Lines ending with punctuation need no full stop
        if line.endswith(('.', '!', '?', ';', ':')):
            return line
        
        # Lines ending with a special symbol (such as +/ or +/-) get a full stop directly
        if line.endswith(('+/', '+/-', '+/-', '±', '~', '...')):
            return line + "."
        
        # Otherwise check whether the next line continues the sentence
        if line and (line[-1].isalpha() or line[-1].isdigit() or not line[-1].isspace()):
            # Check whether the next line continues the sentence
            if next_line and next_line.strip():
                next_line_clean = next_line.strip()
                
                # A next line starting with a lowercase letter, digit or symbol continues the sentence
                if next_line_clean:
                    # Starts with a lowercase letter
                    is_lower_start = next_line_clean[0].islower()
                    # Starts with a digit or symbol (such as #2 or 1)
                    is_special_start = next_line_clean[0].isdigit() or next_line_clean[0] in '#@$%&*()[]{}|\\:;"\'<>?,./'
                    
                    if is_lower_start or is_special_start:
                        # Make sure the next line is not a table row or other non-descriptive content
                        if (not re.match(r'^\d+\s*-\s*', next_line_clean) and 
                            not re.match(r'^[A-Z]\d+\s*-', next_line_clean) and 
                            not re.match(r'^\d{3}\s*-', next_line_clean) and 
                            not re.match(r'^\d+$', next_line_clean) and 
                            not any(keyword in next_line_clean.upper() for keyword in ['A21', 'A24', 'A25', 'A39', 'B.C.'])):
                            # The sentence continues, so no full stop is added
                            return line
            
            # Add a full stop when the sentence does not continue
            return line + "."
        
        return line

    def clean_description(self, text: str) -> str:
        """Remove tables and page headers, then rebuild the description sentences."""
        lines = text.split('\n')
        clean_lines = []
        i = 0
        skip_block = False
        current_sentence = ""
        


        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Start of a page block to skip
            if re.match(r'^===\s*Page\s+\d+\s*===$', line, re.IGNORECASE):
                skip_block = True
                i += 1
                continue

            # Skip the block until the Team Lead line
            if skip_block:
                if re.match(r'^Team Lead:', line, re.IGNORECASE):
                    skip_block = False
                i += 1
                continue

            # Skip page headers and report header information
            if self.is_page_header(line) or self.is_header_info(line):
                i += 1
                continue

            # Skip table titles followed by ELEMENTS DESCRIPTION
            if self.is_table_header(lines, i):
                i += 1
                continue

            # Skip the table header row
            if re.match(r'^ELEMENTS\s+DESCRIPTION\s+UNITS\s+TOTAL\s+CS1\s+CS2\s+CS3\s+CS4$', line, re.IGNORECASE):
                i += 1
                continue

            # Skip section titles (such as SUPERSTRUCTURE or DECK)
            if re.match(r'^(CHANNEL|SUBSTRUCTURE|DECK|SUPERSTRUCTURE|CULVERT|APPROACH|GENERAL|OBSERVATION)$', line, re.IGNORECASE):
                i += 1
                continue

            # Handle B.C. rating lines, including multi-line and title-only entries
            if re.match(r'^[A-Z]\.C\.\d+\s', line):
                # Check whether the line contains the rating in brackets
                if '(' in line and ')' in line:
                    # Complete rating line, so skip it
                    i += 1
                    continue
                elif '(' in line:
                    # The closing bracket is on a later line
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if ')' in next_line:
                            # Closing bracket found, so skip all related lines
                            i = j + 1
                            break
                        elif re.match(r'^[A-Z]\.C\.\d+\s', next_line):
                            # A new B.C. rating line ends the search
                            i += 1
                            break
                        else:
                            j += 1
                    else:
                        # No closing bracket found, so skip the current line
                        i += 1
                    continue
                else:
                    # Title-only rating line, so check the next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # Skip both lines when the next line holds the rating
                        if '(' in next_line and ')' in next_line:
                            i += 2
                            continue
                        # Otherwise skip the title and keep the description on the next line
                        else:
                            i += 1
                            continue
                    else:
                        # Last line, so skip it
                        i += 1
                        continue

            # Keep descriptive text that starts with a number (such as "4 prestress slab units like new")
            if (re.match(r'^\d+\s+[a-z]', line, re.IGNORECASE) and  # Number followed by a word
                not re.match(r'^\d+\s+[A-Z\s]+\s+[A-Z]{1,3}\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+$', line) and  # Not a table row
                len(line.split()) > 3 and  # Several words indicate descriptive text
                # Make sure the line is not a table row with several numbers
                not re.search(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+$', line) and  # Does not end with five numbers
                # Make sure the line is not a rating line
                not re.match(r'^[A-Z]\.C\.\d+\s', line) and  # Not a B.C. rating line
                not re.search(r'Rating\s*\([^)]*\)', line, re.IGNORECASE) and  # Contains no rating information
                # Make sure the line has no unit abbreviations such as LF or EA
                not re.search(r'\b(LF|EA|SF|CY|TON|LB|IN|FT|YD|M|KM|MM|CM|KG|GAL|L|ML|PSI|KSI|MPA|GPA|DEG|RAD|HZ|KHZ|MHZ|V|A|W|KW|HP|RPM|CFM|GPM|CFS|GPD|GPH|GPM|CFH|CFD|GPD|GPH|GPM|CFH|CFD)\b', line, re.IGNORECASE)):  # Contains no common units
                # This is descriptive text, so keep it
                processed_line = self.add_period_to_line(line, lines[i+1] if i+1 < len(lines) else None)
                clean_lines.append(processed_line)
                i += 1
                continue

            # Skip table rows and quantitative ratings
            if (re.match(r'^\d+\s*-\s*', line) or  # Table row starting with a number and a hyphen
                    re.match(r'^[A-Z]\d+\s*-', line) or  # A21 - format
                    re.match(r'^[A-Z]\.C\.\d+\s', line) or  # B.C.05 format (rating line)
                    re.match(r'^\d{3}\s*-', line) or  # 200 - format
                    re.match(r'^\d+$', line) or  # Number only
                    # Table row: number, description, unit and five numbers
                    re.match(r'^\d+\s+[A-Z\s]+\s+[A-Z]{1,3}\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+$', line) or
                    # Remaining parts of rating lines
                    re.search(r'^\s*[A-Z\s]+\s*-\s*[A-Z\s]+\s*Rating\s*\([^)]*\)', line, re.IGNORECASE) or  # Rating title
                    re.search(r'^\s*\([^)]*\)\s*$', line) or  # Rating text in brackets on its own line
                    re.search(r'^\s*[A-Z\s]+\s*-\s*[A-Z\s]+\s*;', line) or  # Rating description ending with a semicolon
                    re.search(r'^\s*[A-Z\s]+\s*-\s*[A-Z\s]+\s*\.\s*\)', line) or  # Rating description ending with a full stop and bracket
                    # Continuation of a rating line
                    re.search(r'^\s*[^)]*\)\s*$', line) or  # Line ending with a bracket
                    # A38 title without a rating
                    re.match(r'^A38\s*-\s*[^(]*$', line) or  # A38 - Deck Drain Condition (no rating in brackets)
                    # Table row with several numbers
                    re.search(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+$', line)):  # Ends with five numbers
                i += 1
                continue

            # Keep a free-text observation written beneath a B.C. rating line
            if (i > 0 and 
                re.match(r'^[A-Z]\.C\.\d+\s', lines[i-1].strip()) and 
                '(' in lines[i-1] and ')' in lines[i-1] and
                not re.match(r'^[A-Z]\.C\.\d+\s', line) and
                not re.search(r'^\s*\([^)]*\)\s*$', line) and
                not re.search(r'Rating\s*\([^)]*\)', line, re.IGNORECASE)):
                # This is a useful description, so keep it
                # Add a full stop before storing it
                processed_line = self.add_period_to_line(line, lines[i+1] if i+1 < len(lines) else None)
                clean_lines.append(processed_line)

                i += 1
                continue



            # Merge sentences that continue across lines
            # A line ending with sentence punctuation completes a sentence
            if line.endswith(('.', '!', '?')):
                # Complete sentence, so store it
                if current_sentence:
                    clean_lines.append(current_sentence + " " + line)
                    current_sentence = ""
                else:
                    clean_lines.append(line)
                i += 1
                continue
            
            # A line ending with a semicolon or colon is part of a sentence
            if line.endswith((';', ':')):
                if current_sentence:
                    current_sentence += " " + line
                else:
                    current_sentence = line
                i += 1
                continue
            
            # A next line starting with a lowercase letter, digit or symbol continues the sentence
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line:
                    # Starts with a lowercase letter
                    is_lower_start = next_line[0].islower()
                    # Starts with a digit or symbol (such as #2 or 1)
                    is_special_start = next_line[0].isdigit() or next_line[0] in '#@$%&*()[]{}|\\:;"\'<>?,./'
                    
                    if is_lower_start or is_special_start:
                        # The sentence continues, so add it to current_sentence
                        if current_sentence:
                            current_sentence += " " + line
                        else:
                            current_sentence = line
                        i += 1
                        continue
            
            # A trailing hyphen means that a word is split across lines
            # Symbols such as +/- are excluded
            if line.endswith('-') and not line.endswith(('+/-', '+/-', '±')):
                # The word is split, so add it to current_sentence
                if current_sentence:
                    current_sentence += " " + line[:-1]  # Remove the hyphen
                else:
                    current_sentence = line[:-1]  # Remove the hyphen
                i += 1
                continue
            
            # Handle complete sentences without a final full stop
            # For example: "Areas of circular cracking along the abutment 1 approach and an area of developing small pot holes along the center line."
            if (line and line[0].isupper() and len(line) > 50 and  # Long sentence
                not line.endswith((';', ':')) and  # Does not end with a semicolon or colon
                'and' in line.lower() and  # Contains a conjunction
                not any(keyword in line.upper() for keyword in ['A21', 'A24', 'A25', 'A39', 'B.C.', '200 -', 'A21 -', 'A24 -', 'A25 -', 'A39 -'])):
                
                # Store any pending sentence first
                if current_sentence:
                    # Add a full stop to the pending sentence
                    if not current_sentence.endswith(('.', '!', '?', ';', ':')):
                        current_sentence += "."
                    clean_lines.append(current_sentence)
                    current_sentence = ""
                
                # Store the current line
                processed_line = self.add_period_to_line(line, lines[i+1] if i+1 < len(lines) else None)
                clean_lines.append(processed_line)
                i += 1
                continue
            
            # Check for other standalone sentences
            # A line starting with a capital letter may be a standalone sentence
            if (line and line[0].isupper() and len(line) > 10 and
                not any(keyword in line.upper() for keyword in ['A21', 'A24', 'A25', 'A39', 'B.C.'])):

                # Store any pending sentence first
                if current_sentence:
                    # Add a full stop to the pending sentence
                    if not current_sentence.endswith(('.', '!', '?', ';', ':')):
                        current_sentence += "."
                    clean_lines.append(current_sentence)
                    current_sentence = ""

                # Store the current line
                processed_line = self.add_period_to_line(line, lines[i+1] if i+1 < len(lines) else None)
                clean_lines.append(processed_line)
                i += 1
                continue
            
            # Handle short standalone sentences (such as "Weep tubes")
            # A short capitalised line is standalone unless the next line starts in lowercase
            if (line and line[0].isupper() and len(line) <= 50 and len(line) > 2 and
                not any(keyword in line.upper() for keyword in ['A21', 'A24', 'A25', 'A39', 'B.C.', 'A38']) and
                not re.match(r'^[A-Z]\d+\s*-', line) and  # Not the A38 - format
                not re.match(r'^\d+\s*-\s*', line) and  # Not the number - format
                not re.match(r'^\d{3}\s*-', line) and  # Not the 200 - format
                # Exclude section titles (such as CHANNEL or DECK)
                not re.match(r'^(CHANNEL|SUBSTRUCTURE|DECK|SUPERSTRUCTURE|CULVERT|APPROACH|GENERAL|OBSERVATION)$', line, re.IGNORECASE)):
                
                # Check whether the next line starts in lowercase
                next_line_lower_start = False
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and next_line[0].islower():
                        next_line_lower_start = True
                
                # Otherwise treat the line as a standalone sentence
                if not next_line_lower_start:
                    # Store any pending sentence first
                    if current_sentence:
                        # Add a full stop to the pending sentence
                        if not current_sentence.endswith(('.', '!', '?', ';', ':')):
                            current_sentence += "."
                        clean_lines.append(current_sentence)
                        current_sentence = ""

                    # Store the current line
                    processed_line = self.add_period_to_line(line, lines[i+1] if i+1 < len(lines) else None)
                    clean_lines.append(processed_line)
                    i += 1
                    continue

            # Default: add the line to current_sentence
            if current_sentence:
                current_sentence += " " + line
            else:
                current_sentence = line
            i += 1

        # Store the remaining sentence
        if current_sentence:
            if not current_sentence.endswith(('.', '!', '?', ';', ':')):
                current_sentence += "."
            clean_lines.append(current_sentence)

        # Join the lines in their original order
        combined_text = ' '.join(clean_lines).strip()
        
        # Remove trailing uppercase words (such as APPROACH or CHANNEL)
        # These are usually section titles
        combined_text = re.sub(r'\s+[A-Z]{2,}\s*\.?\s*$', '', combined_text)
        
        # Fix encoding problems in quotation marks
        combined_text = combined_text.replace('â€"', '"').replace('â€"', '"').replace('â€™', "'")
        
        return combined_text

    def is_table_header(self, lines: List[str], current_index: int) -> bool:
        """Check whether a line is a table title followed by ELEMENTS DESCRIPTION."""
        current_line = lines[current_index].strip()

        # Check whether the line contains a table title keyword
        table_headers = ['Deck', 'Superstructure', 'Substructure', 'Culvert']
        if not any(header.upper() in current_line.upper() for header in table_headers):
            return False

        # Check whether the next line is ELEMENTS DESCRIPTION
        if current_index + 1 < len(lines):
            next_line = lines[current_index + 1].strip()
            if 'ELEMENTS' in next_line.upper() and 'DESCRIPTION' in next_line.upper():
                return True

        return False

    def extract_component_description(self, content: str, component_num: str, component_name: str) -> str:
        """Extract the description of one component."""
        # Find the start of the component section
        start_match = None
        
        # Try the standard name first
        start_pattern = rf'{component_num}\s*-\s*{re.escape(component_name)}'
        start_match = re.search(start_pattern, content, re.IGNORECASE)
        
        # Then try the alternative names
        if not start_match and component_num in self.component_aliases:
            for alias in self.component_aliases[component_num]:
                start_pattern = rf'{component_num}\s*-\s*{re.escape(alias)}'
                start_match = re.search(start_pattern, content, re.IGNORECASE)
                if start_match:
                    break
        
        if not start_match:
            return ""

        start_pos = start_match.end()

        # Find the end of the section (the next component or a marker)
        end_markers = [
            '72 - Approach Roadway Alignment', 
            'GENERAL OBSERVATION', 
            'Channel Profile', 
            '=== Page',  # Page separator
            'Asset #',   # Start of asset information
            'ELEMENTS DESCRIPTION UNITS TOTAL CS1 CS2 CS3 CS4'  # Start of the summary table
        ]
        end_markers.extend([f'{num} - {name}' for num, name in self.components.items() if num != component_num])

        end_pos = len(content)
        for marker in end_markers:
            match = re.search(re.escape(marker), content[start_pos:], re.IGNORECASE)
            if match:
                end_pos = min(end_pos, start_pos + match.start())

        # Extract the description text
        description_text = content[start_pos:end_pos]

        # Remove the rating text in brackets after the component name
        description_text = self.remove_component_rating_description(description_text)

        # Clean the description
        cleaned_description = self.clean_description(description_text)
        
        # Remove trailing uppercase words (such as APPROACH or CHANNEL)
        # These are usually section titles
        cleaned_description = re.sub(r'\s+[A-Z]{2,}\s*\.?\s*$', '', cleaned_description)
        
        return cleaned_description

    def remove_component_rating_description(self, text: str) -> str:
        """Remove the rating text in brackets after the component name."""
        # Match leading text in brackets, such as (8 - VERY GOOD CONDITION - no problems noted.)
        # or (6 - SATISFACTORY CONDITION - structural elements show some minor deterioration.)
        pattern = r'^\s*\([^)]*\)\s*'
        return re.sub(pattern, '', text, flags=re.MULTILINE)

    def process_file(self, file_path: str) -> List[Dict]:
        """Process one report."""
        results = []

        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                return results

        # Extract the basic information
        structure_id = self.extract_structure_id(content)
        source = os.path.basename(file_path).replace('.txt', '')  # Remove the .txt suffix

        # Process each component
        for component_num, component_name in self.components.items():
            # Check whether the component exists, including alternative names
            component_found = False
            if component_num in self.component_aliases:
                # Check all alternative names
                for alias in self.component_aliases[component_num]:
                    component_pattern = rf'{component_num}\s*-\s*{re.escape(alias)}'
                    if re.search(component_pattern, content, re.IGNORECASE):
                        component_found = True
                        break
            else:
                # Check the standard name
                component_pattern = rf'{component_num}\s*-\s*{re.escape(component_name)}'
                component_found = re.search(component_pattern, content, re.IGNORECASE)
            
            if component_found:
                rating = self.extract_component_rating(content, component_num)
                description = self.extract_component_description(content, component_num, component_name)

                # Keep only records with a description and a rating
                if description and rating and rating != 'N/A':
                    # Use "Channel" for all channel records
                    output_component = "Channel" if component_name == "Channel Condition" else component_name
                    
                    results.append({
                        'id': structure_id,
                        'source': source,
                        'description': description,
                        'component': output_component,  # Use the unified component name
                        'rating': rating
                    })

        return results

    def save_to_csv(self, results: List[Dict], output_file: str):
        """Save the records of the target components to a CSV file."""
        if not results:
            print("No records were extracted")
            return

        # Number the records
        for i, result in enumerate(results, 1):
            result['id'] = i

        # Channel and culvert records are not used in this study
        results = [result for result in results if result['component'] in TARGET_COMPONENTS]

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'source', 'description', 'component', 'rating']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved the results to {output_file}")
        print(f"Saved {len(results)} records")

    def process_directory(self, directory_path: str, output_file: str):
        """Process all .txt files in a folder."""
        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} .txt files")

        all_results = []
        for txt_file in txt_files:
            file_path = os.path.join(directory_path, txt_file)
            print(f"Processing {txt_file}")

            results = self.process_file(file_path)
            all_results.extend(results)
            print(f"  Extracted {len(results)} records")

        self.save_to_csv(all_results, output_file)


def main():
    extractor = BridgeInspectionExtractor()

    directory_path = "txt_report"
    output_file = "dataset/vtrans_bridge_inspection_results.csv"
    extractor.process_directory(directory_path, output_file)


if __name__ == "__main__":
    main()