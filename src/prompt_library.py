import pdb

from langchain.prompts import PromptTemplate

class PromptLibrary(object):
    def __init__(self):
        pass

    def snli_write_question(self, label):
        if label == 'contradiction':
            string, string2 = ', that contradict each other', 'contradicts'
        elif label == 'entailment':
            string, string2 = ', where sentence1 entails sentence2', 'entails'
        elif label == 'neutral':
            string, string2 = ', where sentence1 neither entails nor contradicts sentence2', 'is neutral to'
        else: assert False
        pt = PromptTemplate.from_template(
'I will provide you with two sentences sentence1 and sentence2' + string + '. I want you to write a question such that '
'its answer on sentence1 ' + string2 + ' its answer on sentence2.'
'sentence1, and the answer to the query for paragraph2 is sentence2. Adhere to the following guidelines:''''
1. The paragraphs should consist of 4-10 sentences and should typically describe scenes that could be depicted in photographs.
2. The paragraphs should not be written as if they are image captions. For example, they should not reference "a photo", "a scene", or "an image".
3. Similarly, the query should be written as if it is about an image caption. Thus, it should avoid referring to the paragraph as "a photo", "a scene", or "an image".
4. Both paragraphs must address the query you provide.
5. Critically, each sentence should be the most specific answer possible to the query for its respective paragraph. For'''''
' example, suppose sentence1 is "The children are talking to the animals", and '''
'the question you propose is "Who are the children talking to?" The paragraph should not '
'say "The children are talking to giraffes", because "giraffes" are more specific than "animals". '
'In that case, on the question of who the children are talking to, the paragraph should '
'not be more specific than "animals".'
'6. It is crucial that ''''

Fill in the XML tags below. Note that answer1 and answer2 denote the answer to the query for paragraph1 '''
'and paragraph2 respectively.''''

<sentence1>{sent1}</sentence1>
<sentence2>{sent2}</sentence2>
<paragraph1></paragraph1>
<paragraph2></paragraph2>
<query></query>
<answer1>{sent1}</answer1>
<answer2>{sent2}</answer2>''')
        return pt

    def snli_write_all(self, label):
        if label == 'contradiction':
            string = ', which contradict each other'
            string2 = ('Since sentence1 and sentence2 contradict each other, it is crucial that the query be about the contradictory element, and not about '
                       'elements that may be common to the two sentences.')
        elif label == 'entailment':
            string = ', where sentence1 entails sentence2'
            string2 = (
                'Since sentence1 entails sentence2, it is important that the answer to the query for paragraph1 entails the answer to the query for paragraph2.')
        elif label == 'neutral':
            string = ', where sentence1 neither entails nor contradicts sentence2'
            string2 = (
                'Since sentence1 and sentence2 are neutral with respect to each other, '
                'you must be very careful not to write paragraph1 in such a way that sentence2 is definitely true '
                'or definitely false. It must be ambiguous enough that sentence2 may or may not be true based on paragraph1.')
        else: assert False
        pt = PromptTemplate.from_template(
'I will provide you with two sentences sentence1 and sentence2'+string+'. I want you to do the following. '
'Write paragraph1 and paragraph2 and a query, such that the answer to the query for paragraph1 is '
'sentence1, and the answer to the query for paragraph2 is sentence2. The goal is to preserve the relationship between '
'the answers ('+label+'). Adhere to the following guidelines:''''
    1. The paragraphs should consist of 4-10 sentences and should typically describe scenes that could be depicted in photographs.
    2. The paragraphs should not be written as if they are image captions. For example, they should not reference "a photo", "a scene", or "an image".
    3. Similarly, the query should be written as if it is about an image caption. Thus, it should avoid referring to the paragraph as "a photo", "a scene", or "an image".
    4. Both paragraphs must address the query you provide.
    5. Critically, each sentence should be the most specific answer possible to the query for its respective paragraph. For'''''
' example, suppose sentence1 is "The children are talking to the animals", and '''
'the question you propose is "Who are the children talking to?" The paragraph should not '
'say "The children are talking to giraffes", because "giraffes" are more specific than "animals". '
'In that case, on the question of who the children are talking to, the paragraph should '
'not be more specific than "animals".''''
    6. '''+string2+'''

Fill in the XML tags below. Note that answer1 and answer2 denote the answer to the query for paragraph1 '''
'and paragraph2 respectively.''''

<sentence1>{sent1}</sentence1>
<sentence2>{sent2}</sentence2>
<paragraph1></paragraph1>
<paragraph2></paragraph2>
<query></query>
<answer1>{sent1}</answer1>
<answer2>{sent2}</answer2>''')
        return pt

    def snli_answerability(self):
        pt = PromptTemplate.from_template(
'I will provide you with a paragraph and a query. I simply want you to tell me if the paragraph contains an answer to the query. '
'It may be that the answer to the query is complicated, spread over many sentences, or answerable in many ways. '
'In any of these cases you should return "yes". If the query is based on a premise incompatible with the content of the paragraph,'
' or if the query is irrelevant to the content of the paragraph, then you should return "no". If you are not sure, return "maybe".''''

Fill in the XML tags below (put your reasoning in the reasoning tags).

<paragraph>{paragraph}</sentence1>
<query>{query}</query>

Does the paragraph contain an answer to the query?

<reasoning></reasoning>
<answer></answer>''')
        return pt

    def snli_is_def_true(self):
        pt = PromptTemplate.from_template(
'I will provide you with a paragraph and a sentence. I simply want you to tell me if the sentence is DEFINITELY true'
' based on the paragraph. A sentence is definitely true if it follows from the paragraph with little to no inference. '
'If it is DEFINITELY true, then return "yes"; otherwise return "no". If you are unsure, return "maybe".''''

Fill in the XML tags below (put your reasoning in the reasoning tags).

<paragraph>{paragraph}</sentence1>
<sentence>{sentence}</sentence>

Is the sentence DEFINITELY true based on the paragraph?

<reasoning></reasoning>
<answer></answer>''')
        return pt

    def snli_modify_location(self):
        pt = PromptTemplate.from_template(
'I will provide you with a paragraph about a hypothetical image. I simply want you to rewrite the paragraph, changing '
'the location (or setting) to something different. You should otherwise not change the content of the paragraph.''''

For example, if the paragraph is set at the beach, you might change it to a pool. If it is set in a church, you might change it to a mosque.

Importantly, you must make sure that the answer to the question "{query}" is still "{sentence}".

Fill in the XML tags below.

<paragraph>{paragraph}</paragraph>

Now change the setting of the paragraph, while making sure that the answer to the question "{query}" is still "{sentence}".

<modified_paragraph></modified_paragraph>''')
        return pt

    def snli_is_about_location(self):
        pt = PromptTemplate.from_template(
'I will provide you with a question about a hypothetical photograph. I simply want you to tell me if question is '
'about the location, or setting, of the photo. If the question is about the location, then you should '
'return "yes"; otherwise return "no". If you are unsure, return "maybe".''''

For example, if the question is "Where are the children standing?" or "What is the setting of the image?", return "yes".
If the question is "Who is talking with the woman?" or "What is the man holding?", return "no".

Fill in the XML tags below (put your reasoning in the reasoning tags).

<question>{query}</question>

Is the query asking about a location or setting?

<reasoning></reasoning>
<answer></answer>''')
        return pt

    def snli_is_an_answer(self):
        pt = PromptTemplate.from_template(
'Pretend you are looking at a photograph depicting a scene, and someone asks you a question about it.'
' I will provide you a sentence. I want you to tell me if the sentence would be a possible direct answer to the question '
'for the hypothetical photograph. Since this is based on a photograph, you may not have all the information about the '
'scene depicted; therefore, an answer may be plausible even if it is not extremely specific or detailed. The most important thing'
' is that it provides a direct answer to the question, and if it does you should return "yes". Otherwise, return "no". If you are not sure, return "maybe". ''''

Fill in the XML tags below (put your reasoning in the reasoning tags).

<question>{query}</question>
<sentence>{answer}</sentence>

Is the sentence a possible answer to the question?

<reasoning></reasoning>
<response></response>''')
        return pt

    def snli_write_paragraph(self):
        pt = PromptTemplate.from_template(
'I will to provide you with a sentence, and I want you to write a short paragraph and a '
'question about the paragraph, such that the answer to the question is this sentence. However, '
'this sentence should be the most specific answer possible to the question given the paragraph.''''

For example, suppose the answer is “The children are talking to the animals”, and '''
'the question you propose is “Who are the children talking to?” The paragraph should not '
'say “The children are talking to giraffes”, because that is more specific than the answer. '
'Thus, in this case, on the question of who the children are talking to, the paragraph should '
'not be more specific than “animals”.''''

Please fill in the XML tags below.

<sentence>{sentence}</sentence>
<paragraph></paragraph>
<question></question>
<answer>{sentence}</answer>''')
        return pt

    def snli_write_paragraph_with_question(self):
        pt = PromptTemplate.from_template(
'I will to provide you with a sentence and a question. I want you to write a short paragraph and a '
'such that the answer to the question is this sentence. However, '
'this sentence should be the most specific answer possible to the question given the paragraph.''''

For example, suppose the answer is “The children are talking to the animals”, and '''
'the question is “Who are the children talking to?” The paragraph should not '
'say “The children are talking to giraffes”, because that is more specific than the answer. '
'Thus, in this case, on the question of who the children are talking to, the paragraph should '
'not be more specific than “animals”.''''

Please fill in the XML tags below.

<sentence>{sentence}</sentence>
<paragraph></paragraph>
<question>{question}</question>
<answer>{sentence}</answer>''')
        return pt

    def paraphrase_with_question_and_answer(self):
        pt = PromptTemplate.from_template(
'I will provide you with a document, a question, and the answer to the question given the document.'
" Your task is to paraphrase the document such that it does not look like it's written by the same person."
' This means that you can change the order the information is presented and the number of sentences, but'
' you should not change the content. Most importantly, you must ensure that the answer to the question on'
' the paraphrased document is the same as the answer to the'
' question on the original document. Fill in the XML tags below.''''

<document>{document}</document>

<question>{question}</question>

<answer>{answer}</answer>

<paraphrased_doc></paraphrased_doc>''')
        return pt

    def robustqa_get_label(self):
        pt = PromptTemplate.from_template(
'I will provide you with a question, and an answer which either provides a positive ("yes") or negative ("no") opinion. '
'I simply want you to tell me whether the opinion is "yes" or "no". '
'A clear giveaway would be if the answer itself has these labels in it, especially if the answer starts with one of '
'these labels. Fill in the reasoning and response XML tags below - the response must be "yes" or "no".''''

<question>{query}</question>
<answer>{answer}</answer>

<reasoning></reasoning>
<response></response>

''')
        return pt

    def factscore_write_question(self):
        pt = PromptTemplate.from_template(
'I will provide you with an atomic fact. I want you to write a simple question such that it could be answered '
'by the fact. For example, if the fact is "The sky is blue.", then the question would be "What color is the sky?"'
' Fill in the XML tags below, including your reasoning.''''

<fact>{fact}</fact>

<reasoning></reasoning>

<question></question>''')
        return pt

    def ragtruth_entire_relevant(self):
        pt = PromptTemplate.from_template(
'I will provide you with a query and a response to the query written by an LLM. Note that the query is about a specific document,'
' but you do not need to know the contents of that document to perform the following task. I just want you to tell me if '
'the response contains an answer to the query. Because you do not have the document, you won\'t be able to tell if the answer is '
'correct or not, but that is not relevant. You just need to tell me if the passage I provide you contains a potential answser to the query.''''

For example, if the query is "What is the store's name?", and the passage contains a sentence like "The store's name is Portle", then you should return "yes". Otherwise, return "no".

Fill in the XML tags for reasoning and answer below. Your answer must be "yes" or "no".

<query>{query}</query>
<response>{response}</response>

Does the response contain an answer to the query?

<reasoning></reasoning>
<answer></answer>''')
        return pt

    def ragtruth_snippet_relevant(self):
        pt = PromptTemplate.from_template(
'I will provide you with a query and a response to the query written by an LLM. Note that the query is about a specific document,'
' but you do not need to know the contents of that document to perform the following task. I also will provide you with a snippet from the response.'
' I just want you to tell me if snippet is a potential answer to the query; the rest of the response is simply provided as context. '
'Because you do not have the underlying document, you won\'t be able to tell if the snippet is '
'a correct correct or not, but that is not what you are being asked. You just need to tell me if the snippet is a potential answer to the query.''''

For example, if the query is "What is the name of the store in Mabileen?", and the snippet is "The store's name is Portle", then '''
'you should return "yes". If the snippet is something like "There is no store in Mabileen", you should return "yes". However, if the '
'snippet is "Mabileen has 100 residents", you should return "no", since that is not an answer to the query.''''

Fill in the XML tags for reasoning and answer below. Your answer must be "yes" or "no".

<query>{query}</query>
<response>{response}</response>
<snippet>{snippet}</snippet>

Is the snippet a potential answer to the query?

<reasoning></reasoning>
<answer></answer>''')
        return pt

    def task_answer_question_full_sentence(self):
        pt = PromptTemplate.from_template(
'You are a reliable question-answering system. I will provide you with a document and a question. I simply want you to provide an answer '
'the question based on the document using one or more full sentences. It is very important that your answer be based only on the document:'
' do not apply any knowledge beyond what is contained in the document. Be very concise: you do not need to provide more information '
'than that which answers the question.''''

Fill in the resasoning and answer XML tags.

<document>{document}</document>
<question>{query}</question>

<reasoning></reasoning>
<answer></answer>

''')
        return pt

    def task(self, label_set, prompt_type, use_query, dataset):
        entailment_def = "entailment: Return 'entailment' only if the answer to the query for document2 (the hypothesis) is necessarily true given the answer to the query for document1 (the premise)."
        not_entail_def = "not_entailment: Return 'not_entailment' only if the answer to the query for document2 (the hypothesis) does not entail the answer to the query for document1 (the premise)."
        contradiction_def = "contradiction: Return 'contradiction' only if the answer to the query for document2 (the hypothesis) is necessarily false given the answer to the query for document1 (the premise)."
        not_contradiction_def = "not_contradiction: : Return 'not_contradiction' only if the answer to the query for document2 (the hypothesis) is not necessarily false given the answer to the query for document1 (the premise) – in other words, document1 does not contradict document2."
        neutral_def = "neutral: Return 'neutral' only if the answer to the query for document2 (the hypothesis) is neither entailed nor contradicted given the answer to the query for document1 (the premise)."
        if label_set == ['entailment', 'not_entailment']:
            label_str = '"entailment" or "not_entailment"'
            label_str2 = 'entail or not entail'
            defs = '\t– '+entailment_def + '\n\t– ' + not_entail_def
        elif label_set == ['entailment', 'contradiction']:
            label_str = '"entailment" or "contradiction"'
            label_str2 = 'entail or contradict'
            defs = '\t– ' + entailment_def + '\n\t– ' + contradiction_def
        elif label_set == ['entailment', 'contradiction', 'neutral']:
            label_str = '"entailment", "contradiction", or "neutral"'
            label_str2 = 'entail, contradict, or is it neutral to'
            defs = '\t– ' + entailment_def + '\n\t– ' + contradiction_def + '\n\t– ' + neutral_def
        elif label_set == ['contradiction', 'not_contradiction']:
            label_str = '"contradiction" or "not_contradiction"'
            label_str2 = 'contradiction or not contradict'
            defs = '\t– ' + contradiction_def + '\n\t– ' + not_contradiction_def
        else: assert False

        if prompt_type == 'few':
            allow_multiple_queries = 'factscore' in dataset
            if not allow_multiple_queries:
                extra_template = ('''

<example_{label}>
<document1>{doc1}</document1>

<document2>{doc2}</document2>

<query>{query}</query>

Now, does document1 ''' + label_str2 + ''' document2 with respect to the query?'''
                                       ' First, provide your reasoning in the reasoning tags. Then, provide your answer (' + label_str + ') in the answer tags. Make sure your final answer is one of these answer choices.''''

<reasoning>{reasoning}</reasoning>

<answer>{label}</answer>
</example_{label}>''')
            else: # allows mutliple queries for same document pair
                extra_template = ('''

<example_{label}>
<query>{query}</query>

Now, does document1 ''' + label_str2 + ''' document2 with respect to the query?'''
                                       ' First, provide your reasoning in the reasoning tags. Then, provide your answer (' + label_str + ') in the answer tags. Make sure your final answer is one of these answer choices.''''

<reasoning>{reasoning}</reasoning>

<answer>{label}</answer>
</example_{label}>''')
            if dataset == 'snli':
                if len(label_set) == 2 and 'entailment' in label_set and 'not_entailment' in label_set:
                    c, n = 'not_entailment', 'not_entailment'
                else: c, n = 'contradiction', 'neutral'
                examples = [
                    ('entailment', {
                        'doc1': "Outside the bustling kitchen of a local restaurant, a lone figure stands in the shadows, his white shirt a stark contrast to the grime of the alley. He takes a long drag from a cigarette, the smoke curling around him as he leans against the dumpster, seemingly lost in his own thoughts.",
                        'doc2': "The man, dressed in a white restaurant shirt, steps outside the building to take a break from his work. He lights a cigarette and inhales deeply, enjoying the brief respite from the bustling kitchen. The fresh air and change of scenery provide a moment of solace as he collects his thoughts before returning to his duties inside.",
                        'query': "Where is the man in the white restaurant shirt smoking?",
                        'reasoning': "document1 states that the man is smoking outside the bustling kitchen. document2 states that the man is smoking outside, enjoying \"the brief respite from the bustling kitchen\". Therefore, in both cases the man is smoking outside of a bustling kitchen, and thus the first document entails the second with respect to this query. Thus, the correct label is \"entailment\"."
                    }),
                    (c, {
                        'doc1': "In a dimly lit studio, an art student sits focused on their latest project. With a steady hand, they grasp a vibrant blue Sharpie and begin to meticulously draw a captivating pattern across the surface of a thick, white poster board. The rhythmic strokes of the marker create a mesmerizing design, as the student's creative vision takes shape on the blank canvas before them.",
                        'doc2': "The art student is working on a creative project in their studio. They are focused on their work, carefully manipulating the materials to create a unique and intricate pattern. The student's hands move skillfully, weaving the fibers together to shape the scarf taking form.",
                        'query': "What materials is the art student using to create their pattern?",
                        'reasoning': "document1 states that the art student is creating a pattern with a blue Sharpie. document2 states that the student is weaving fibers to make a scarf. Using a Sharpie to make artwork contradicts weaving fibers to make a scarf. Therefore, the document is contradictory with respect to the materials being used. Thus, the correct label is \""+c+"\"."
                    }),
                    (n, {
                        'doc1': "The man stood atop the hill, his gaze fixed on the vast expanse before him. With a pair of binoculars pressed against his eyes, he carefully scanned the landscape, taking in the breathtaking scenery that stretched out as far as the eye could see. The gentle breeze rustled the leaves of the trees, and the sun's warm rays cast a golden glow over the rolling hills and distant mountains.",
                        'doc2': "The man is sitting in a comfortable armchair, his eyes intently focused on the pages of the book he is holding. The room is quiet, with the only sound being the occasional turn of a page as the man becomes absorbed in the words before him.",
                        'query': "What is the man doing with the binoculars?",
                        'reasoning': "document1 suggests that a man is looking at a breathtaking landscape with binoculars. In document2, the man does not have any binoculars. Therefore, document1 neither entails nor contradicts document2 with respect to what the man is doing with binoculars. Thus, the correct label is \""+n+"\"."
                    }),
                ]
                string = ''.join([extra_template.format(label=k, doc1=v['doc1'], doc2=v['doc2'],reasoning=v['reasoning'], query=v['query']) for k,v in examples])
            elif dataset == 'ragtruth':
                assert len(label_set) == 2 and 'entailment' in label_set and 'not_entailment' in label_set
                c, n = 'not_entailment', 'not_entailment'
                examples = [
                    ('entailment', {
                        'doc1': "passage 1:For people with Type 1 diabetes, however, having measurable amounts of ketones in the urine or blood is cause for concern. Ketones in a person with Type 1 diabetes may be a sign that his diabetes is out of control, he is ill or has an infection, or he is under extreme stress.\n\npassage 2:Iron-deficiency anemia is the most common type of anemia. It happens when you do not have enough iron in your body. Iron deficiency is usually due to blood loss but may occasionally be due to poor absorption of iron. Pregnancy and childbirth consume a great deal of iron and thus can result in pregnancy-related anemia.\n\npassage 3:Usually, menstrual bleeding lasts about 4 to 5 days and the amount of blood lost is small (2 to 3 tablespoons). However, women who have menorrhagia usually bleed for more than 7 days and lose twice as much blood. If you have bleeding that lasts longer than 7 days per period, or is so heavy that you have to change your pad or tampon nearly every hour, you need to talk with your doctor.\n\n",
                        'doc2': "Concern about blood loss should occur in situations such as iron-deficiency anemia, which can be caused by blood loss, and in cases of menorrhagia where women bleed for more than 7 days per period or have to change their pad or tampon nearly every hour.",
                        'query': "when to be concerned about blood loss",
                        'reasoning': "We must make sure that every fact related to when to be worried about blood lost stated by document2 is backed up by document1. The first statement says to be concerend about blood lost in situations of iron-defeciency anemia. That is indeed supported by passage 2. The second statement makes various claims about menorrhagia, saying that if any of them happen you should be concerned about blood loss. Every single one of these claims is backed up by passage 3. Therefore, the correct label is \"entailment\"."
                    }),
                    (c, {
                        'doc1': "passage 1:Roasting times for a turkey breast cooked in a 325 degrees F oven: Unstuffed, a 2 to 3-pound turkey will cook from 1 1/2 to 2 hours. Unstuffed, a 4 to 6-pound turkey will cook from 1 1/2 to 2 1/4 hours.Unstuffed, a 7 to 8-pound turkey will cook from 2 1/4 to 3 1/4 hours; if stuffed, it will cook in 3 to 4 hours.Roasted Turkey Breast Variations: Lemon and Thyme: Season the butter with 3 tablespoons fresh thyme leaves, 1 teaspoon lemon juice, and 1 teaspoon lemon zest.et the meat breast-side up in the pan. Roast for about 2 hours or until an instant read thermometer inserted into the thickest part of the breast registers 160 degrees F. Transfer the turkey to a cutting board and tent loosely with foil for 20 minutes.\n\npassage 2:Rating Newest Oldest. Best Answer: 2 pound fresh (or frozen and thawed) turkey breast Kosher salt and ground pepper Dried herbs and spices of choice: sage, thyme, garlic powder, onion powder Vegetable oil Brush the turkey breast with vegetable oil.Sprinkle liberally with salt, pepper, herbs, and spices.elevance. Rating Newest Oldest. Best Answer: 2 pound fresh (or frozen and thawed) turkey breast Kosher salt and ground pepper Dried herbs and spices of choice: sage, thyme, garlic powder, onion powder Vegetable oil Brush the turkey breast with vegetable oil. Sprinkle liberally with salt, pepper, herbs, and spices.\n\npassage 3:1 Heat the oil in the fryer and put the turkey breast into the oil for frying. 2  Cook for 4 to 5 minutes per pound. 3  For example, a 5 pound (2.26 kg) turkey breast will cook for 20 to 25 minutes. Use an outdoor propane fryer or an indoor electric fryer. 2  Heat the oil in the fryer and put the turkey breast into the oil for frying. 3  Cook for 4 to 5 minutes per pound. 4  For example, a 5 pound (2.26 kg) turkey breast will cook for 20 to 25 minutes.\n\n",
                        'doc2': "Based on the given passages, it is difficult to determine the exact time required to cook a 2 lb turkey breast. However, according to passage 1, an unstuffed 2 to 3-pound turkey will cook from 1 1/2 to 2 hours. Passage 2 suggests roasting a 2 lb turkey breast at 325 degrees F for about 2 hours or until an instant read thermometer inserted into the thickest part of the breast registers 160 degrees F. Passage 3 does mention cooking a 2 lb turkey breast in a fryer for about 10 minutes per pound, but this method is not recommended as it can result in overcooking and drying out the meat. Therefore, the best option would be to roast the 2 lb turkey breast in an oven at 325 degrees F for approximately 2 hours or until fully cooked.",
                        'query': "how long to cook 2lb turkey breast",
                        'reasoning': "document2 says \"Passage 2 suggests roasting a 2 lb turkey breast at 325 degrees F for about 2 hours or until an instant read thermometer inserted into the thickest part of the breast registers 160 degrees F.\" However, Pasasge 1 says that. Therefore, the information about how to cook a 2lb turkey breast in document2 is not backed up by document1. Thus, the correct label is \"" + c + "\"."
                    }),
                    (n, {
                        'doc1': "passage 1:If so, apply for the appropriate license. 5. Contact your insurance agent and inquire about insurance for your new mobile pet-grooming service. 1  You should insure the van and its contents. 2  You may also wish to purchase liability insurance in case you have a mishap while grooming a dog or cat. Visit veterinarians in your area and tell them about your mobile grooming van. 2  Ask independent pet supply stores if you can leave some flyers for their customers. 3  You could even offer to set up the van in their parking lot and offer grooming specials to their customers.\n\npassage 2:1 A mobile grooming service is a grooming shop on wheels. 2  Therefore, you need to retrofit the van with a grooming table, a tub, a water supply tank and pump, a generator and secure storage for your grooming supplies. Visit veterinarians in your area and tell them about your mobile grooming van. 2  Ask independent pet supply stores if you can leave some flyers for their customers. 3  You could even offer to set up the van in their parking lot and offer grooming specials to their customers.\n\npassage 3:Make a living as a pet groomer. The dog image by Beausoir from Fotolia.com. A pet grooming business is a good way to earn a part-time or full-time income. If you enjoy working with animals and have the patience to give them thorough care, you may find a pet grooming business rewarding and profitable.tep 5. Purchase supplies and equipment for your pet grooming business. You may need a variety of grooming tables, tubs, dryers and cages. You may also need combs, brushes, scissors, clippers, shampoos and styling tools.\n\n",
                        'doc2': "To start up a mobile dog grooming business, you will need to follow these steps:\n\n1. Apply for the appropriate license.\n2. Contact your insurance agent and inquire about insurance for your new mobile pet-grooming service. Insure the van and its contents, as well as liability insurance in case you have a mishap while grooming a dog or cat.\n3. Retrofit the van with a grooming table, a tub, a water supply tank and pump, a generator and secure storage for your grooming supplies.\n4. Visit veterinarians in your area and tell them about your mobile grooming van.\n5. Ask independent pet supply stores if you can leave some flyers for their customers.\n6. Offer to set up the van in their parking lot and offer grooming specials to their customers.\n7. Purchase supplies and equipment for your pet grooming business, including grooming tables, tubs, dryers and cages, combs, brushes, scissors, clippers, shampoos and styling tools.\n8. Start your business and build a loyal customer base by providing excellent service and maintaining high standards of cleanliness and safety.",
                        'query': "how to start up a mobile dog grooming business",
                        'reasoning': "document 2 says \"Start your business and build a loyal customer base by providing excellent service and maintaining high standards.\". However, this is not mentioned by document1. Thus, the correct label is \"" + n + "\"."
                    }),
                ]
                string = ''.join([extra_template.format(label=k, doc1=v['doc1'], doc2=v['doc2'],
                                                        reasoning=v['reasoning'], query=v['query']) for k, v in
                                  examples])
            elif dataset == 'robustqa':
                assert 'contradiction' in label_set and 'not_contradiction' in label_set
                examples = [
                    ('contradiction', {
                        'doc1': "No, by itself, a zip code is not PII. It's demographic data, but since many thousands of people may live in a single zip code, there's nothing about the zip code alone that would identify which individual(s) it represents. In this case, a zip code may also represent multiple clients, if there are multiple clients of the firm in a single zip code. So, almost by definition, it can't be PII, since it doesn't uniquely identify a single individual.",
                        'doc2': "Yes, of course it is personally identifying information. It provides identifying information about a person, so why on earth might it be considered otherwise? Consider a shopkeeper in a small town saying \"I think I shall invest in [very unpopular company] when I get home tonight.\" His customer says \"if you do, I will never shop here again!\" That night, the customer sees that another investor appears on the company's map for that zipcode. Would you consider it unreasonable of the customer to stop shopping at that store? Would you consider it unreasonable of others, on hearing this tale, to also stop shopping there? Remember, small town, there probably weren't any investors at all from there before. Would you consider it reasonable of the shopkeeper to then sue the company for leaking his private investment information and hence causing damage to his business? So the potential number of zipcodes where the combination of [is a user of that webside's service] and [lives in that zipcode] and [when they started investing] will be uniquely identifying is obviously pretty huge. But it's worse than that. The following zipcodes have a population of exactly one person: 05141, 67843, 88264, 98222, 99790. There are over a hundred zipcodes with populations under 10. 11109 has an area of just two city blocks. If you live in 38639 you are also black. If you live in 02562 you are white (better than 99% probability for both). If you live in Beverly Hills 90210, you are rich and everyone knows it. If you live in 90209, you are still rich, but likely have a chip on your shoulder about your zipcode being less famous. There are a little under 8 billion of us. That means that we need only \"33 bits of entropy\" -- that is, 33 yes/no questions which slice the population roughly in half, like \"are you male\", \"do you live outside China/India\", etc -- to identify any individual. A zipcode provides between 16 bits of information (the two most populous zipcodes have over 110,000 people in) and the full 33 (those 6 zipcodes above). That is to say, a zipcode alone is at least half the information needed to uniquely identify anybody. [Edit: and of course, in the US, businesses are persons. Many, MANY companies with a technical population of zero have their own zipcodes. If that company invests in another company, they may well NOT like that information being put up publicly.] [Edit2: Zip codes are explicitly called out as PII by Massachusetts (https://casetext.com/case/tyler-v-michaels-stores) and California (http://scocal.stanford.edu/opinion/pineda-v-williams-sonoma-33947).]",
                        'query': "is a zip code considered pii?",
                        'reasoning': "document1 says clearly responds \"No\", while document2 says \"yes, of course\". Therefore, these documents are contradictory with respect to this query; thus, the correct label is \"contradiction\"."
                    }),
                    ('not_contradiction', {
                        'doc1': "No, by itself, a zip code is not PII. It's demographic data, but since many thousands of people may live in a single zip code, there's nothing about the zip code alone that would identify which individual(s) it represents. In this case, a zip code may also represent multiple clients, if there are multiple clients of the firm in a single zip code. So, almost by definition, it can't be PII, since it doesn't uniquely identify a single individual.",
                        'doc2': "By itself, no. You can't identify an individual by just knowing that persons zipcode. Zipcode is merely demographic information. But, you might be able to combine a large number of individual pieces of demographics to identify someone. Zipcode + Age + Sex + Income might easily be enough to identify someone. If I told you that Person A was male, 60 years old, lived in zip code 98039, and had an income of 2 billion dollars last year, you might guess I was talking about Bill Gates. (I have no idea how much Bill made last year, but I'm trying to illustrate a point). The point being that the aggregate of individually non-PII demographic information can itself become PII.",
                        'query': "is a zip code considered pii?",
                        'reasoning': "Both of these documents clearly provide a \"no\" answer. Therefore the two documents entail one another with respect to this quesiton. Thus, the correct label is \"not_contradiction\"."
                    })
                ]
                string = ''.join([extra_template.format(label=k, doc1=v['doc1'], doc2=v['doc2'],
                                                        reasoning=v['reasoning'], query=v['query']) for k, v in
                                  examples])
            elif dataset in ('factscore_chatgpt', 'factscore_chatgpt_all'):
                # NOTE THE ALL CASE IS TECHNICALLY CHEATING A LITTLE BIT!!!!
                assert len(label_set)==2 and 'entailment' in label_set and 'not_entailment' in label_set
                c, n = 'not_entailment', 'not_entailment'
                examples = [
                    {
                        'doc1': "<s>Manuel Salazar (footballer) Manuel Alejandro Salazar Rivas (born January 23, 1986 in San Salvador) is a retired Salvadoran professional footballer. He represented the El Salvador.</s><s>Club career. Salazar started his career at Salvadoran Second Division side Telecom before joining Luís Ángel Firpo in 2005. In December 2008, Salazar along with his club and national team teammate Carlos Monteagudo, both were invited to train with Mexican Primera División A team Monterrey 1a. A (effectively CF Monterreys reserve team), with the hopes of signing with the club's second division side. After the two-week trial, both players returned to El Salvador, with the promise from head coach Gerardo Jiménez, that he would look to open negotiations to transfer them to the club, whether on a full-time basis, or on loan however neither player was ever signed. On July 1, 2010, Salazar signed a six-month contract with FAS only to rejoin Firpo for the 2011 Clausura claiming FAS owing him three months wages.</s><s>International career. Salazar received his first call up to the national team in January 2007 and made his debut for El Salvador in####SPECIAL####SEPARATOR#### a February 2007 UNCAF Nations Cup match against Nicaragua. He earned a total of 50 caps, scoring no goals, and has represented his country in 18 FIFA World Cup qualification matches. He has played at the 2007 UNCAF Nations Cup, as well as at the 2007 and 2009 CONCACAF Gold Cups. His final international game was an October 2010 friendly match against Costa Rica.</s><s>Retirement from football. On May 18, 2011, Salazar decided to retire from the sport due to long-standing knee problems and to focus on his academic and business ventures (he owns a restaurant).</s>",
                        'doc2': "Manuel Salazar is a retired Peruvian footballer who played as a forward. He was born on August 5, 1975, in Lima, Peru. Salazar began his professional career with Sporting Cristal in 1994 and played for the club until 1999, winning the Peruvian Primera Division title twice during his time there.\nIn 2000, Salazar joined Alianza Lima, where he played for three seasons and won another Primera Division title. He then had a brief stint with Universitario de Deportes in 2003 before moving to Mexico to play for Club Necaxa.\nAfter playing for several other Mexican clubs, Salazar returned to Peru in 2008 to play for Cienciano, where he helped the team win the 2008 Copa Sudamericana. He retired from professional football in 2010.\nSalazar also played for the Peru national team, earning 28 caps and scoring six goals between 1995 and 2003.",
                    },
                    ('entailment', {
                        'query': "What was Manuel Salazar's occupation?",
                        'reasoning': "document1 states that Salazar was a \"professional footballer\". document2 states that he is a retired \"footballer\". Therefore, document1 entails document2 with respect to Salazar's profession."
                    }),
                    ('not_entailment', {
                        'query': "In what year did he have a stint with Universitario de Deportes?",
                        'reasoning': "document1 does not mention \"Universitario de Deportes\". document2 states that he had a stint there in 2003. Therefore, since one document does not contain information about this query, document1 does not entail document2 with respect to this query."
                    }),
                    ('not_entailment', {
                        'query': "When was he born?",
                        'reasoning': "document1 states that Salazar was born on January 23, 1986. document2 states that he was born on August 5, 1975. Therefore, document1 contradicts document2 with resect to this query, and hence the relationshi is \"not_entailment\"."
                    })
                ]
                string = '''
<documents>
<document1>{doc1}</document1>

<document2>{doc2}</document2>
</documents>'''.format(doc1=examples[0]['doc1'], doc2=examples[0]['doc2'])
                string += ''.join([extra_template.format(label=k,
                                                        reasoning=v['reasoning'], query=v['query']) for k, v in
                                  examples[1:]])
            elif dataset in ('factscore_instructgpt', 'factscore_instructgpt_all'):
                # NOTE THE ALL CASE IS TECHNICALLY CHEATING A LITTLE BIT!!!!
                assert len(label_set) == 2 and 'entailment' in label_set and 'not_entailment' in label_set
                c, n = 'not_entailment', 'not_entailment'
                examples = [
                    {
                        'doc1': "<s>Manuel Salazar (footballer) Manuel Alejandro Salazar Rivas (born January 23, 1986 in San Salvador) is a retired Salvadoran professional footballer. He represented the El Salvador.</s><s>Club career. Salazar started his career at Salvadoran Second Division side Telecom before joining Luís Ángel Firpo in 2005. In December 2008, Salazar along with his club and national team teammate Carlos Monteagudo, both were invited to train with Mexican Primera División A team Monterrey 1a. A (effectively CF Monterreys reserve team), with the hopes of signing with the club's second division side. After the two-week trial, both players returned to El Salvador, with the promise from head coach Gerardo Jiménez, that he would look to open negotiations to transfer them to the club, whether on a full-time basis, or on loan however neither player was ever signed. On July 1, 2010, Salazar signed a six-month contract with FAS only to rejoin Firpo for the 2011 Clausura claiming FAS owing him three months wages.</s><s>International career. Salazar received his first call up to the national team in January 2007 and made his debut for El Salvador in####SPECIAL####SEPARATOR#### a February 2007 UNCAF Nations Cup match against Nicaragua. He earned a total of 50 caps, scoring no goals, and has represented his country in 18 FIFA World Cup qualification matches. He has played at the 2007 UNCAF Nations Cup, as well as at the 2007 and 2009 CONCACAF Gold Cups. His final international game was an October 2010 friendly match against Costa Rica.</s><s>Retirement from football. On May 18, 2011, Salazar decided to retire from the sport due to long-standing knee problems and to focus on his academic and business ventures (he owns a restaurant).</s>",
                        'doc2': "Manuel Salazar is a professional footballer from Mexico. He currently plays for Liga MX side Club Tijuana as a defender. He began his career with Club Tijuana in 2017, making his debut in the Liga MX in April of that year. He has since established himself as an important part of the team's defensive line, helping the team to their first ever Liga MX title in 2019. Salazar is a strong, physical defender with good passing and tackling abilities. He is also an excellent reader of the game and an effective communicator on the pitch.",
                    },
                    ('entailment', {
                        'query': "What is Manuel Salazar's profession?",
                        'reasoning': "document1 states that Salazar was a \"professional footballer\". document2 states that he is a \"footballer\". Therefore, document1 entails document2 with respect to Salazar's profession."
                    }),
                    ('not_entailment', {
                        'query': "What team does he currently play for?",
                        'reasoning': "document1 states that Salazar is a \"retired\" footballer: hence he does not play for a team. document2 states that he \"currently plays for Liga MX side Club Tijuana\". Therefore, document1 directly contradicts document2 with respect to this query, so the label is \"not_entailment\"."
                    }),
                    ('not_entailment', {
                        'query': "How good is Salazar's tackling ability?",
                        'reasoning': "document1 does not discuss Salazar's tackling ability at all. document2 states that he has \"good passing and tackling abilities\". Therefore, since document1 does not contain information about this query, it does not entail document2 with resect to this query. Therefore the label is \"not_entailment\"."
                    })
                ]
                string = '''
            <documents>
            <document1>{doc1}</document1>

            <document2>{doc2}</document2>
            </documents>'''.format(doc1=examples[0]['doc1'], doc2=examples[0]['doc2'])
                string += ''.join([extra_template.format(label=k,
                                                         reasoning=v['reasoning'], query=v['query']) for k, v in
                                   examples[1:]])
            elif dataset in ('factscore_perplexityai', 'factscore_perplexityai_all'):
                # NOTE THE ALL CASE IS TECHNICALLY CHEATING A LITTLE BIT!!!!
                assert len(label_set) == 2 and 'entailment' in label_set and 'not_entailment' in label_set
                c, n = 'not_entailment', 'not_entailment'
                examples = [
                    {
                        'doc1': "<s>Fernando (footballer, born 1984) Fernando de Jesus Ribeiro (born 7 July 1984), more commonly known as Fernando is a Brazilian football goalkeeper, who currently plays for Pars Jonoubi in the Persian Gulf Pro League.</s><s>Club career.</s><s>Club career.:Esteghlal Khuzestan. Fernando signed with Persian Gulf Pro League club Esteghlal Khuzestan in the summer transfer window of 2015. He impressed in the first half of the 2015–16 season, keeping the most clean sheets in the first 15 games of the season. On 13 May 2016, with the help of several saves from Fernando, Esteghlal Khuzestan won their last game of the season against Zob Ahan and became the champions of the Persian Gulf Pro League for the first time in their history.</s><s>Club career.:Gostaresh Foolad. Fernando joined Tabrizi side Gostaresh Foolad in the summer of 2016 after being named goalkeeper of the season in the Persian Gulf Pro League.</s><s>Honours. - Esteghlal Khuzestan - Persian Gulf Pro League (1): 2015–16</s><s>Honours.:Individual. - Persian Gulf Pro League Team of the####SPECIAL####SEPARATOR#### Year: 2015–16 - Persian Gulf Pro League Most clean sheets: 2015–16 - IRFF Awards Goalkeeper of the Year: 2015–16</s>",
                        'doc2': "Fernando Lucas Martins (born March 3, 1992), more commonly known as Fernando, is a Brazilian footballer who plays as defensive midfielder for Antalyaspor[1]. He signed a five-year contract with Shakhtar Donetsk in 2013 for a transfer fee of €11 million[1]. Fernando was part of the Brazil national team that played in the 2013 FIFA Confederations Cup on home soil[1].\nThere are three other Brazilian footballers named Fernando. Fernando Francisco Reges (born July 25, 1987) plays for Sevilla as a defensive midfielder[2]. Fernando Almeida de Oliveira (born June 18, 1978), known as just Fernando, currently plays for Vitória as a midfielder[3]. Fernandinho (born May 4, 1985) is regarded as one of the best defensive midfielders of his generation and has captained the Brazil national team on many occasions. He currently plays for Athletico Paranaense as a midfielder[4].",
                    },
                    ('entailment', {
                        'query': "What is the nationality of Fernando Lucas Martins?",
                        'reasoning': "Both documents state that Martins is a \"Brazilian\" footballer. Therefore, document1 entails document2 with respect to Salazar's nationality."
                    }),
                    ('not_entailment', {
                        'query': "When was Fernando Lucas Martins born?",
                        'reasoning': "document1 states that Martins was born in \"1984\". document2 states that he was born \"March 3, 1992\". Therefore, document1 directly contradicts document2 with respect to when Martins was born, so the label is \"not_entailment\"."
                    }),
                    ('not_entailment', {
                        'query': "How much was the transfer fee?",
                        'reasoning': "document1 does not say anything about a transfer fee. document2 states that the transfer fee was 11 million euros. Therefore, since document1 does not contain information about this query, it does not entail document2 with resect to this query. Therefore the label is \"not_entailment\"."
                    })
                ]
                string = '''
                       <documents>
                       <document1>{doc1}</document1>

                       <document2>{doc2}</document2>
                       </documents>'''.format(doc1=examples[0]['doc1'], doc2=examples[0]['doc2'])
                string += ''.join([extra_template.format(label=k,
                                                         reasoning=v['reasoning'], query=v['query']) for k, v in
                                   examples[1:]])
            else: assert False
            if not allow_multiple_queries:
                fewshot = "\n\nI will provide you with a few examples:"+string+"\n\nNow it's your turn!"
            else: # allows same doc pair
                assert len(label_set) == 2
                fewshot = "\n\nI will provide you with a few examples, each based on the same pair of documents: this shows that the relationship can be \""+label_set[0]+"\" for some queries, and \""+label_set[1]+"\" for others.\n" + string + "\n\nNow it's your turn!"
        elif prompt_type == 'zero':
            fewshot = ''
        elif prompt_type == 'qanli':
            pt = PromptTemplate.from_template(
'I will provide you with two documents (possibly each just a single sentence). Treat document1 as the premise and'
' document2 as the hypothesis. I want you to tell me how document1 relates to document2. Specifically, you should'
' return one of the following NLI labels: '+ label_str + '.''''

The labels are defined as follows:
''' + defs +'''

<document1>{doc1}</document1>

<document2>{doc2}</document2>

Now, does document1 '''+label_str2+''' document2?'''
' First, provide your reasoning in the reasoning tags. Then, provide your answer ('+label_str +') in the answer tags. Make sure your final answer is one of these answer choices.''''

<reasoning></reasoning>

<answer></answer>''')
            return pt
        else: assert False

        pt = PromptTemplate.from_template(
'I will provide you with two documents (document1 and document2) as well as a query about the'
' documents. Treat document1 as the premise and document2 as the hypothesis. I want you to'
' tell me how document1 relates to document2 with respect to the query. Specifically, you should return'
' one of the following NLI labels: '+label_str+'. You can assume'
' that both documents contain the complete information needed to address the query.''''

The labels are defined as follows:
''' + defs + fewshot+'''

<document1>{doc1}</document1>

<document2>{doc2}</document2>

<query>{query}</query>

Now, does document1 '''+label_str2+''' document2 with respect to the query?''' +(' Do your best if no query is provided.' if not use_query else '') +''
' First, provide your reasoning in the reasoning tags. Then, provide your answer ('+label_str +') in the answer tags. Make sure your final answer is one of these answer choices.''''

<reasoning></reasoning>

<answer></answer>''')
        return pt