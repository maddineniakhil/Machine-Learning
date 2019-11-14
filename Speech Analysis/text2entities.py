from google.cloud import language_v1
from google.cloud.language_v1 import enums
from google.oauth2 import service_account
from nltk.tokenize import word_tokenize
import pandas as pd
from flask import Flask, request
import nltk

app = Flask(__name__)

@app.route('/', methods=['POST'])
def akhil():
    req_data = request.get_json()
    text_content = req_data['text']
    print(text_content)

    def download_nltk():
        try:
            nltk.download('punkt')
        except EOFError:
            return
    download_nltk()

    # text_content = "Hey Jade, I have just spent $40 at starbucks"

    def find_industry(organization):
        organization = organization.lower()
        print(organization)
        company_data = pd.read_csv('C:/Users/akhil/Downloads/company_data.csv')
        industry = company_data.loc[company_data['name'] == organization]
        return industry.iloc[0]['industry']

    def text_similarity(Y):
        Command1 = "Add funds to category spent"
        Command2 = "Set budget to category limit"

        Command1_list = word_tokenize(Command1)
        Command2_list = word_tokenize(Command2)
        Y_list = word_tokenize(Y)

        temp1 = Command1_list + Y_list
        temp2 = Command2_list + Y_list

        l1 = []
        l2 = []
        for w in temp1:
            if w in Command1_list:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_list:
                l2.append(1)
            else:
                l2.append(0)
        c = 0
        # cosine formula
        for i in range(len(temp1)):
            c += l1[i] * l2[i]
        cosine1 = c / float((sum(l1) * sum(l2)) ** 0.5)

        # Comparing similarity with second command
        Command2_list = word_tokenize(Command2)

        l1 = []
        l2 = []
        for w in temp2:
            if w in Command2_list:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_list:
                l2.append(1)
            else:
                l2.append(0)
        c = 0
        # cosine formula
        for i in range(len(temp2)):
            c += l1[i] * l2[i]
        cosine2 = c / float((sum(l1) * sum(l2)) ** 0.5)

        if (cosine1 > cosine2):
            action = 1
        else:
            action = 2
        print(cosine1, cosine2)
        return action

    industry = "None"

    def sample_analyze_entities(text_content):

        organization = "None"
        price = 0
        """
        Analyzing Entities in a String

        Args:
          text_content The text content to analyze
        """
        credentials = service_account.Credentials.from_service_account_file('C:/Users/akhil/Downloads/HackGT-4c5034201e49.json')
        client = language_v1.LanguageServiceClient(credentials=credentials)

        type_ = enums.Document.Type.PLAIN_TEXT
        language = "en"
        document = {"content": text_content, "type": type_, "language": language}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = enums.EncodingType.UTF8

        response = client.analyze_entities(document, encoding_type=encoding_type)
        # Loop through entitites returned from the API
        for entity in response.entities:
            for metadata_name, metadata_value in entity.metadata.items():
                # print(u"{}: {}".format(metadata_name, metadata_value))
                if enums.Entity.Type(entity.type).name == "ORGANIZATION":
                    organization = entity.name
                elif enums.Entity.Type(entity.type).name == "PRICE":
                    price = entity.metadata['value']
        return organization, price

    organization, price = sample_analyze_entities(text_content)
    if(organization != "None"):
        industry = find_industry(organization)
    action = text_similarity(text_content)
    return {"price": price, "action": action, "industry": industry}

if __name__ == "__main__":
    app.run(debug=True)