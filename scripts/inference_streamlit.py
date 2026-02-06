import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import joblib
import nltk
from src.utils.constants import ROOT_DIR
from src.data.preprocessing import word2features

@st.cache_resource
def load_model():
    model_path = ROOT_DIR / "models" / "crf_model.pkl"
    
    if not os.path.isfile(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    
    try:
        crf_model = joblib.load(model_path)
        return crf_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def tokenize_and_tag(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    tree = nltk.ne_chunk(pos_tags, binary=False)
    
    chunk_tags = []
    for item in tree:
        if hasattr(item, 'label'):
            for word, pos in item:
                chunk_tags.append('B-NP')
        else:
            chunk_tags.append('O')
    
    pos_label_names = ['#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 
                       'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 
                       'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 
                       'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
    
    chunk_label_names = ['O', 'B-ADJP', 'I-ADJP', 'B-ADVP', 'I-ADVP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 
                         'I-INTJ', 'B-LST', 'I-LST', 'B-NP', 'I-NP', 'B-PP', 'I-PP', 'B-PRT', 'I-PRT', 
                         'B-SBAR', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-VP', 'I-VP']
    
    sentence_dict = {
        'tokens': tokens,
        'pos_tags': [pos_label_names.index(pos) if pos in pos_label_names else 0 for word, pos in pos_tags],
        'chunk_tags': [chunk_label_names.index(chunk) if chunk in chunk_label_names else 0 for chunk in chunk_tags]
    }
    
    return sentence_dict

def extract_entities(tokens, labels):
    entities = []
    current_entity = []
    current_label = None
    
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append((' '.join(current_entity), current_label))
            current_entity = [token]
            current_label = label[2:]
        elif label.startswith('I-') and current_label == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((' '.join(current_entity), current_label))
            current_entity = []
            current_label = None
    
    if current_entity:
        entities.append((' '.join(current_entity), current_label))
    
    return entities

def highlight_entities(text, tokens, labels):
    entity_colors = {
        'PER': '#FFB6C1',
        'ORG': '#87CEEB', 
        'LOC': '#90EE90',
        'MISC': '#FFD700'
    }
    
    entity_icons = {
        'PER': '👤',
        'ORG': '🏢',
        'LOC': '📍',
        'MISC': '🔹'
    }
    
    result = []
    i = 0
    
    for token, label in zip(tokens, labels):
        if label != 'O':
            entity_type = label.split('-')[1] if '-' in label else label
            color = entity_colors.get(entity_type, '#CCCCCC')
            icon = entity_icons.get(entity_type, '•')
            result.append(f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; margin: 2px;">{icon} {token} <sub>{entity_type}</sub></span>')
        else:
            result.append(token)
        i += 1
    
    return ' '.join(result)

def main():
    try:
        st.set_page_config(
            page_title="NER CoNLL-2003 Classifier", 
            page_icon="🏷️",
            layout="wide"
        )
    except:
        pass
    
    crf_model = load_model()
    
    if crf_model is None:
        st.stop()
    
    st.markdown("# 🏷️ Named Entity Recognition Classifier")
    st.markdown("**Powered by Conditional Random Fields (CRF)**")
    
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This model identifies and classifies named entities:
        - 👤 **PER** (Person)
        - 🏢 **ORG** (Organization)
        - 📍 **LOC** (Location)
        - 🔹 **MISC** (Miscellaneous)
        """)
        
        st.markdown("## 📊 Model Info")
        st.markdown("""
        - **Dataset**: CoNLL-2003
        - **Algorithm**: CRF (sklearn-crfsuite)
        - **Test F1-Score**: 83.38%
        - **Features**: Word shapes, POS tags, chunk tags, context
        """)
        
        st.markdown("## 🔗 Links")
        st.markdown("📄 [CoNLL-2003 Dataset](https://huggingface.co/datasets/eriktks/conll2003)")
        st.markdown("🔬 [CRF Research](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)")
        st.markdown("[📂 GitHub Repository](https://github.com/ahmedxnov/NER-CoNLL2003)")
        st.markdown("---")
        st.markdown("**👨‍💻 Developer:** [Ahmad Khaled](https://www.linkedin.com/in/ahmad-khaled-hamed/)")
    
    st.markdown("## 🔍 Entity Recognition")
    st.markdown("Enter text below to extract named entities:")
    
    text_input = st.text_area(
        "Enter text for entity recognition:", 
        height=150, 
        placeholder="Example: Apple Inc. was founded by Steve Jobs in Cupertino, California."
    )
    
    predict_button = st.button("🎯 Extract Entities", type="primary")
    
    if predict_button and text_input and text_input.strip():
        try:
            sentence_dict = tokenize_and_tag(text_input)
            tokens = sentence_dict['tokens']
            
            pos_label_names = ['#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 
                               'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 
                               'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 
                               'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
            
            chunk_label_names = ['O', 'B-ADJP', 'I-ADJP', 'B-ADVP', 'I-ADVP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 
                                 'I-INTJ', 'B-LST', 'I-LST', 'B-NP', 'I-NP', 'B-PP', 'I-PP', 'B-PRT', 'I-PRT', 
                                 'B-SBAR', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-VP', 'I-VP']
            
            features = [word2features(sentence_dict, j, pos_label_names, chunk_label_names) 
                       for j in range(len(tokens))]
            
            predictions = crf_model.predict([features])[0]
            
            st.divider()
            st.markdown("## 🎯 Results")
            
            entities = extract_entities(tokens, predictions)
            
            if entities:
                st.markdown("**Annotated Text:**")
                highlighted = highlight_entities(text_input, tokens, predictions)
                st.markdown(highlighted, unsafe_allow_html=True)
                
                st.markdown("**Extracted Entities:**")
                
                entity_groups = {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}
                for entity, label in entities:
                    if label in entity_groups:
                        entity_groups[label].append(entity)
                
                cols = st.columns(4)
                icons = {'PER': '👤', 'ORG': '🏢', 'LOC': '📍', 'MISC': '🔹'}
                labels_full = {'PER': 'Persons', 'ORG': 'Organizations', 'LOC': 'Locations', 'MISC': 'Miscellaneous'}
                
                for i, (label, icon) in enumerate(icons.items()):
                    with cols[i]:
                        st.markdown(f"**{icon} {labels_full[label]}**")
                        if entity_groups[label]:
                            for entity in entity_groups[label]:
                                st.markdown(f"- {entity}")
                        else:
                            st.markdown("_None found_")
            else:
                st.info("No named entities detected in the text.")
            
            with st.expander("📋 Model Information & Limitations"):
                st.info("""
                **Model Training:** This CRF model was trained on CoNLL-2003 news articles.
                
                **Best Performance:** Input text similar to news articles for optimal results.
                
                **Limitations:**
                - Requires proper grammar and punctuation
                - Performance may vary on non-news text
                - Entity boundaries may be imperfect on unusual text formats
                - Depends on NLTK's POS tagging accuracy
                """)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.code(traceback.format_exc())
    elif predict_button:
        st.error("Please enter some text for entity recognition.")

if __name__ == "__main__":
    main()
