from lxml import etree # lxml library is for processing and parsing XML files

def parse_tei_xml(file_path):
    try:
        tree = etree.parse(file_path)
        root = tree.getroot() # the root element is refering to <TEI>

        # Initialize lists to store results
        plain_text = []
        rhetorical_devices = []
        lexical_segments = []

        # Iterate over all <seg> elements and returns rhetorical or lexis
        for seg in root.xpath('.//seg'):
            seg_type = seg.get('type')
            seg_subtype = seg.get('subtype', 'No subtype')  # Default to 'No subtype' if missing
            seg_text = seg.text

            # If text exists, process it
            if seg_text:
                plain_text.append(seg_text.strip())

                # Separate by type (rhetorical or lexical)
                if seg_type == 'rhet':
                    rhetorical_devices.append((seg_text.strip(), seg_subtype))
                elif seg_type == 'lexis':
                    lexical_segments.append((seg_text.strip(), seg_subtype))

        # Join and return the results
        return {
            "text": " ".join(plain_text),
            "rhetorical_devices": rhetorical_devices,
            "lexical_tags": lexical_segments
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    
if __name__ == "__main__":
    # List of XML file paths
    file_paths = [
        "/Users/Admin/AI-ML-Camp/AI-ML-Camp/data/xml/Vita1119.xml", 
        "/Users/Admin/AI-ML-Camp/AI-ML-Camp/data/xml/Arboleda1119.xml", 
        "/Users/Admin/AI-ML-Camp/AI-ML-Camp/data/xml/Memorias Final.xml"
    ]
    
    # Parse each file and collect the results
    for file_path in file_paths:
        result = parse_tei_xml(file_path)

        if result:
            print(f"\n--- Full Extracted Text from {file_path} ---\n")
            print(result["text"][:1000])  # Print first 1000 characters

            print("\n--- Rhetorical Devices ---")
            if result["rhetorical_devices"]:
                for r in result["rhetorical_devices"][:5]:  # Show first 5
                    print(f"Type: {r[1]}, Text: {r[0]}")
            else:
                print("No rhetorical devices found.")

            print("\n--- Lexical Tags ---")
            if result["lexical_tags"]:
                for l in result["lexical_tags"][:5]:  # Show first 5
                    print(f"Type: {l[1]}, Text: {l[0]}")
            else:
                print("No lexical tags found.")
        else:
            print(f"Skipping file {file_path} due to errors.")
