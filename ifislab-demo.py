import streamlit as st
from PIL import Image
import json


from ifis_sumup import hl1, hl2


st.markdown('<style>b{background-color: yellow;}</style>', unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)

image1 = Image.open('logotype_neuronys_couleur_horizontal.png')
st.sidebar.image(image1, use_column_width=True)
image2 = Image.open('ifislab-logo.png')
st.image(image2, width=400,caption="Evaluation de différentes méthodes d'auto-surlignage")

verb = st.sidebar.selectbox(
    'Sélectionnez un algo:',
    ('LEXRANK', 'CLUSTERING'))

if (verb == 'LEXRANK'):
    st.subheader("Surlignage avec l'algorithme LEXRANK")
    st.header('')

    example_text= """La Commission européenne a annoncé vendredi faire appel de l'arrêt des juges de l'UE qui avaient annulé sa décision exigeant d'Apple le remboursement à l'Irlande de 13 milliards d'euros d'avantages fiscaux que Bruxelles jugeait indus. La Commission européenne riposte dans le litige qui l'oppose à Apple. La pomme de discorde : 13 milliards d'euros d'avantages fiscaux que Bruxelles considère comme indus à Dublin. Selon la Commission européenne, Apple a rapatrié entre 2003 et 2014 l'ensemble des revenus engrangés en Europe en Irlande pour y bénéficier d'un traitement fiscal favorable et échapper ainsi à la quasi totalité des impôts, grâce à un accord passé avec les autorités de Dublin. En juillet dernier le Tribunal de l'UE avait donné raison au géant américain, ce qui constituait un sérieux revers pour la vice-présidente de la Commission européenne, Margrethe Vestager, chargée de la Concurrence. Mais cette dernière ne désarme pas. Bruxelles annonce ce vendredi faire appel. "La bataille entre Apple, l'Irlande et l'UE est longue, une saga sans fin", analyse Darren McCaffrey, correspondant d'euronews à Bruxelles_._ "Mais la Commission européenne à Bruxelles estime qu'il s'agit d'une bataille importante. Elle considère que le jugement de juillet dernier crée un mauvais précédent qui la mettrait dans une position désavantageuse pour de futurs cas, lorsqu'il s'agira de s'attaquer aux grands géants américains", souligne-t-il. "C'est également une affaire politiquement importante pour Margrethe Vestager. Elle est la commissaire européenne chargée de la concurrence et elle souhaite que l'Union européenne s'affirme davantage face aux grands groupes afin qu'ils ne puissent échapper à l'impôt", poursuit-il. La Commission européenne va donc porter l'affaire devant la Cour européenne de Justice et assure que le Tribunal de l'UE a commis un "certain nombre d'erreurs de droit". De son côté, Apple assure avoir respecté la loi en Irlande et surtout l'entreprise de Tim Cook continue de bénéficier du soutien officiel de l'Irlande. Le ministre irlandais des Finances Paschal Donohoe a souligné dans un communiqué que, selon Dublin, Apple avait payé "un montant correct" d'impôts. "Le processus de cet appel pourrait durer jusqu'à deux ans", a-t-il ajouté. Pendant ce temps, les 13 milliards d'euros se retrouvent aujourd'hui sur un compte gelé, en attendant une décision finale de la justice européenne. Ce qui risque d'être encore long."""
    paragraph=''
    paragraph = st.text_area("Texte à surligner: (Cliquer sur RUN pour lancer l'analyse)", example_text)
    ratio = st.number_input('Ratio de surlignage ? (0.4 means 40%):',step=0.05,min_value=0.1,max_value=1.0,value=0.4)
    if (st.button("Run")):
        with st.spinner('En cours ...'):
            ret = hl1(paragraph, ratio)
            st.subheader('Texte initial')
            st.write(ret["raw_text"], unsafe_allow_html=True)
            st.subheader('Surlignage proposé:')
            st.write(ret["highlighted_html"], unsafe_allow_html=True)
            st.subheader('JSON retourné:')
            st.json(json.dumps(ret))

if (verb == 'CLUSTERING'):
    st.subheader("Surlignage avec l'algorithme CLUSTERING")
    st.header('')

    example_text= """La Commission européenne a annoncé vendredi faire appel de l'arrêt des juges de l'UE qui avaient annulé sa décision exigeant d'Apple le remboursement à l'Irlande de 13 milliards d'euros d'avantages fiscaux que Bruxelles jugeait indus. La Commission européenne riposte dans le litige qui l'oppose à Apple. La pomme de discorde : 13 milliards d'euros d'avantages fiscaux que Bruxelles considère comme indus à Dublin. Selon la Commission européenne, Apple a rapatrié entre 2003 et 2014 l'ensemble des revenus engrangés en Europe en Irlande pour y bénéficier d'un traitement fiscal favorable et échapper ainsi à la quasi totalité des impôts, grâce à un accord passé avec les autorités de Dublin. En juillet dernier le Tribunal de l'UE avait donné raison au géant américain, ce qui constituait un sérieux revers pour la vice-présidente de la Commission européenne, Margrethe Vestager, chargée de la Concurrence. Mais cette dernière ne désarme pas. Bruxelles annonce ce vendredi faire appel. "La bataille entre Apple, l'Irlande et l'UE est longue, une saga sans fin", analyse Darren McCaffrey, correspondant d'euronews à Bruxelles_._ "Mais la Commission européenne à Bruxelles estime qu'il s'agit d'une bataille importante. Elle considère que le jugement de juillet dernier crée un mauvais précédent qui la mettrait dans une position désavantageuse pour de futurs cas, lorsqu'il s'agira de s'attaquer aux grands géants américains", souligne-t-il. "C'est également une affaire politiquement importante pour Margrethe Vestager. Elle est la commissaire européenne chargée de la concurrence et elle souhaite que l'Union européenne s'affirme davantage face aux grands groupes afin qu'ils ne puissent échapper à l'impôt", poursuit-il. La Commission européenne va donc porter l'affaire devant la Cour européenne de Justice et assure que le Tribunal de l'UE a commis un "certain nombre d'erreurs de droit". De son côté, Apple assure avoir respecté la loi en Irlande et surtout l'entreprise de Tim Cook continue de bénéficier du soutien officiel de l'Irlande. Le ministre irlandais des Finances Paschal Donohoe a souligné dans un communiqué que, selon Dublin, Apple avait payé "un montant correct" d'impôts. "Le processus de cet appel pourrait durer jusqu'à deux ans", a-t-il ajouté. Pendant ce temps, les 13 milliards d'euros se retrouvent aujourd'hui sur un compte gelé, en attendant une décision finale de la justice européenne. Ce qui risque d'être encore long."""
    paragraph=''
    paragraph = st.text_area("Texte à surligner: (Cliquer sur RUN pour lancer l'analyse)", example_text)
    ratio = st.number_input('Ratio de surlignage ? (0.4 means 40%):',step=0.05,min_value=0.1,max_value=1.0,value=0.4)
    nb_clusters = st.number_input('Nombre de clusters ? (0 pour calcul automatique):',step=1,min_value=0,value=0)
    if (st.button("Run")):
        with st.spinner('En cours ...'):
            ret = hl2(paragraph, ratio, nb_clusters)
            st.subheader('Texte initial')
            st.write(ret["raw_text"], unsafe_allow_html=True)
            st.subheader('Surlignage proposé:')
            st.write(ret["highlighted_html"], unsafe_allow_html=True)
            st.subheader('JSON retourné:')
            st.json(json.dumps(ret))
