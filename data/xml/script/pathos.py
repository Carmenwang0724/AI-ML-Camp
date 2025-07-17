#I am using the Gemini API and Prompt engineering to identify pathos
import google.generativeai as genai

#API Key and model version
genai.configure(api_key="AIzaSyC8tDpO3DJG5yA9dgq7L2_UrY9B25oThWU")
model = genai.GenerativeModel("models/gemini-1.5-flash")

#Function that tells Gemini what to do (include training examples)
def extract_pathos_segments_gemini(document_text):
    prompt = f"""
You are an expert in rhetorical analysis.

The following text is written in Spanish or Catalan and may contain multiple examples of emotional appeals (pathos). 
Your task is to:

1. Read the full text.
2. Identify all specific segments that use emotional language to appeal to the reader's feelings.
3. Extract and list each of those emotional segments (in their original language).
4. For each, explain briefly in English why it is an emotional appeal.

Here are a few examples of emotional text:

1. "el qual conpuso Teresa de Cartajena seyendo apasyonada de graues dolençias"
   This an example of pathos.

2. "Salid, señores, yveréis la más desventurada, desamparada y más maldita mujer del mundo"
   This is an example of pathos.

3. "Y yo le dije, con muchas lágrimas"
   This is an example of pathos.

4. “Los que morauan en tinieblas y en sonbra de muerte, luz les es demostrada.”
   This is an example of pathos

Now please tell me if the following sentences are pathos or not:
\"\"\"{document_text}\"\"\"
"""
#Ask to generate response
    response = model.generate_content(prompt)
    return response.text.strip()


#Load in the document text
if __name__ == "__main__":
    document = """
 [G]rand tienpo ha, virtuosa señora, que la niebla de tristeza tenporal e humana cubrió
los términos de mi beuir e con vn espeso toruellino de angustiosas pasyones me lleuó a vna
!nsula que se llama "Oprobrium hominum et abiecio plebis" donde tantos años ha que en
ella biuo, si vida llamar se puede, jamás pude yo ver persona que endereçase mis pies por
la carrera de paz, nin me mostrase camino por donde pudiese llegar a poblado de plazeres.
As! que en este exillyo e tenebroso destierro, más sepultada que morada me sintiendo,
plogo a la misericordia del muy Alt!simo alunbrarme con la luçerna de su piadosa graçia,
porque pudiese poner mi nonbre en la nómina de aquellos de quien es escrito: “Los que
morauan en tinieblas y en sonbra de muerte, luz les es demostrada.
    """

 # Call the function and print the output
result = extract_pathos_segments_gemini(document)
print(result)
