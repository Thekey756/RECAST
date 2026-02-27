# Length
# During validation, check if the number of words in the response is within 20% of the target word count
word_length_template_around = [f'Use around {{}} words.',
                               f'Limit your response to approximately {{}} words.',
                               f'Aim for around {{}} words in your answer.',
                               f'Keep your answer close to {{}} words.',
                               f'Provide a detailed response of approximately {{}} words.',
                               f'Your answer should be about {{}} words, plus or minus 20%.',
                               f'Aim for approximately {{}} words',
                               f'Keep your answer around {{}} words',
                               ]

# During validation, check if the number of words in the response is below the specified word count
word_length_template_below = [f'Keep the answer under {{}} words.',
                              f'no more than {{}} words.',
                              f'Do not exceed {{}} words.',
                              f'Strictly limit the answer to {{}} words.',
                              f'Stay within {{}} words.',
                              f'{{}} words maximum.',
                              f'Use fewer than {{}} words.',
                              f'Cap your response at {{}} words.',
                              f'Answer in {{}} words or less.',
                              ]

# During validation, check if the number of words in the response is between the specified word counts
word_length_template_with_2 = [f'Limit the response to {{}}-{{}} words.',
                               f'Respond in roughly {{}} to {{}} words.',
                               f'Target a response between {{}} and {{}} words.',
                               f'Answer in approximately {{}}–{{}} words.',
                               f'Keep the response between {{}} and {{}} words.',
                               f'Aim for {{}} to {{}} words in your reply.',
                               f'Limit your answer to a length of {{}}–{{}} words.',
                               f'Adhere to a word count of {{}} to {{}}.',
                               ]

word_length_template_group = [word_length_template_around, word_length_template_below, word_length_template_with_2]


sentence_length_template_exact = [f'Provide exactly {{}} sentences in your answer.',
                                  f'Use exactly {{}} sentences in your response.',
                                  f'Your response must contain exactly {{}} sentences.',
                                  f'Strictly use {{}} sentences in your answer.',
                                  f'Adhere to a limit of exactly {{}} sentences.',
                                  f'Structure your answer in exactly {{}} sentences.',
                                  f'Craft a {{}}-sentence response.',
                                  f'The answer shall comprise exactly {{}} sentences.']
sentence_length_template_around = [f'Aim for approximately {{}} sentences (±2).',
                                   f'Your answer should be around {{}} sentences, give or take a few.',
                                   f'Target around {{}} sentences in your response.',
                                   f'Keep your answer to roughly {{}} sentences.',
                                   f'Respond with approximately {{}} sentences.',
                                   f'The response should consist of approximately {{}} sentences.']
sentence_length_template_below = [f'Limit your response to {{}} sentences.',
                                  f'use no more than {{}} sentences.',
                                  f'Do not exceed {{}} sentences in your response.',
                                  f'Cap your reply at {{}} sentences.',
                                  f'Stay within {{}} sentences.',
                                  f'Adhere to a maximum of {{}} sentences.']
sentence_length_template_between = [f'Keep your answer to {{}}–{{}} sentences.',
                                    f'Respond in {{}} to {{}} sentences.',
                                    f'Provide a response of {{}}–{{}} sentences.',
                                    f'Aim for a response between {{}} and {{}} sentences.',
                                    f'Keep your reply within the range of {{}}–{{}} sentences.',
                                    f'The response should comprise {{}}–{{}} sentences.',
                                    f'Maintain a sentence count between {{}} and {{}}.',
                                    f'Provide an answer consisting of roughly {{}} to {{}} sentences.']
sentence_length_template_group = [sentence_length_template_below,
                                  sentence_length_template_around,
                                  sentence_length_template_below,
                                  sentence_length_template_around,]

# Format

format_template = [f'Respond in "{{}}" format.', f'Format your answer as valid "{{}}".', f'Provide the output strictly in "{{}}" format.']

# Keyword
keyword_template = [f'Include the keywords "{{}}" {{}} times in your response.',    # key num
                    f'Your response must feature "{{}}" {{}} times.',               # key num
                    f'Use "{{}}" {{}} times when responding.',                      # key num
                    f'Ensure your answer contains {{}} "{{}}".',                    # num key
                    f'Incorporate {{}} "{{}}" into your answer.']                  # num key

start_with_template = [f'Begin your response with the word "{{}}".', f'"Start your answer with "{{}}".', f'Open your reply using "{{}}" as the first word.']
end_with_template = [f'End your response with the word "{{}}".', f'Make sure the last word of your reply is "{{}}".', f'Your response must terminate with "{{}}".']

#Uppercase & Lowercase
all_upper_template = ['Write your entire response in UPPERCASE.', 'Use ONLY CAPITAL LETTERS in your answer.', 'Type everything in CAPS LOCK.']
all_lower_template = ['Write your entire response in lowercase.', 'Use only small letters in your answer.', 'Avoid any capital letters in your reply.']

#No Commas
no_commas_template = ['Do not use any commas in your response.', 'Avoid commas entirely in your answer.', 'Exclude all commas from your output.']



