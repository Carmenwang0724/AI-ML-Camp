import pandas as pd
from lxml import etree
import spacy

# Function to parse the XML files and extract rhetorical devices
def parse_tei_xml(file_path):
    try:
        # Parse the XML file
        tree = etree.parse(file_path)
        root = tree.getroot()  # Get the root of the XML document

        # Initialize lists to store metaphorical and non-metaphorical devices
        metaphor_devices = []
        non_metaphor_devices = []

        # Iterate through all the <seg> elements and check for type and subtype
        for seg in root.xpath('.//seg'):
            seg_type = seg.get('type')
            seg_subtype = seg.get('subtype', 'No subtype')  # Default to 'No subtype' if missing
            
            # If the segment is a metaphor, concatenate the text across line breaks
            full_text = []
            if seg_type == 'rhetoric':
                for text in seg.itertext():
                    full_text.append(text.strip())
                
                # Combine the texts, ensuring that the content is continuous across line breaks
                metaphor_text = ' '.join(full_text).strip()

                if metaphor_text:
                    if seg_subtype.lower() == 'metaphor':
                        metaphor_devices.append(metaphor_text)  # Metaphor (1)
                    else:
                        non_metaphor_devices.append(metaphor_text)  # Non-metaphor (0)

        return metaphor_devices, non_metaphor_devices

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], []

# Function to preprocess text (tokenization, lemmatization, etc.)
def preprocess_text(text):
    # Ensure the input is a valid string
    if not isinstance(text, str):
        return {
            "tokens": [],
            "lemmas": [],
            "pos_tags": [],
            "named_entities": []
        }

    # Load the pre-trained Spanish spaCy model
    nlp = spacy.load("es_core_news_sm")

    # Step 2: Tokenization and Processing with SpaCy
    doc = nlp(text)

    # Step 3: Lemmatization, POS, and Dependency Parsing
    processed_tokens = []
    for token in doc:
        processed_tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "tag": token.tag_,
            "ent_type": token.ent_type_,
        })

    # Step 4: Named Entity Recognition (NER)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {
        "tokens": processed_tokens,
        "lemmas": [token['lemma'] for token in processed_tokens],
        "pos_tags": [token['pos'] for token in processed_tokens],
        "named_entities": named_entities
    }

# Load your original metaphor dataset (CSV) and extract fragments
csv_file_path = 'metaphors.csv'  # Adjust the file path as needed
df = pd.read_csv("/Users/Admin/AI-ML-Camp/AI-ML-Camp/metaphor.csv", encoding='utf-8')

# Extract metaphor fragments from CSV
fragments = df['Fragment'].dropna().tolist()

# List of XML files to process
file_paths = [
    "/Users/Admin/AI-ML-Camp/AI-ML-Camp/data/xml/Vita1119.xml", 
    "/Users/Admin/AI-ML-Camp/AI-ML-Camp/data/xml/Arboleda1119.xml", 
    "/Users/Admin/AI-ML-Camp/AI-ML-Camp/data/xml/Memorias Final.xml"
]

# Store metaphorical and non-metaphorical devices from all XML files
all_metaphors = []
all_non_metaphors = []

# Parse each XML file and collect metaphorical and non-metaphorical devices
for file_path in file_paths:
    metaphor_devices, non_metaphor_devices = parse_tei_xml(file_path)
    all_metaphors.extend(metaphor_devices)
    all_non_metaphors.extend(non_metaphor_devices)

# Combine metaphorical devices from CSV and XML
all_metaphors_combined = fragments + all_metaphors
all_non_metaphors_combined = all_non_metaphors

# Preprocess metaphorical and non-metaphorical examples
processed_data = []

# Process metaphorical examples (label as 1)
for metaphor in all_metaphors_combined:
    result = preprocess_text(metaphor)
    processed_data.append({
        "text": metaphor,
        "tokens": result["tokens"],
        "lemmas": result["lemmas"],
        "pos_tags": result["pos_tags"],
        "named_entities": result["named_entities"],
        "label": 1  # 1 for metaphor
    })

# Process non-metaphorical examples (label as 0)
for non_metaphor in all_non_metaphors_combined:
    result = preprocess_text(non_metaphor)
    processed_data.append({
        "text": non_metaphor,
        "tokens": result["tokens"],
        "lemmas": result["lemmas"],
        "pos_tags": result["pos_tags"],
        "named_entities": result["named_entities"],
        "label": 0  # 0 for non-metaphor
    })

# Convert processed data into a DataFrame
processed_df = pd.DataFrame(processed_data)

# Save the combined and processed data to a new CSV file
processed_df.to_csv("combined_metaphors_nonmetaphors.csv", index=False)

print("Processing complete! The data has been saved to 'combined_metaphors_nonmetaphors.csv'.")
