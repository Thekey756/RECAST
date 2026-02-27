import os
import json
import re
import csv
import time
import argparse
import logging
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import rule-based evaluation components
from util import (
    evaluate_word_length, evaluate_sentence_length, evaluate_keyword,
    evaluate_format, evaluate_start_with, evaluate_end_with,
    check_english_uppercase, check_english_lowercase, contains_no_punctuation
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
DICTION = {'Length':['word_length', 'sentence_length'],
     'Format':'format',
     'Keyword':'keyword',
     'Strat_With':'start_with',
     'End_With':'end_with',
     'All_Upper':'all_upper',
     'All_lower':'all_lower',
     'No_Commas':'no_commas',
     }
# Evaluation prompts extracted from the documentation
EVALUATION_PROMPT = """
Please act as an objective and fair evaluator to analyze the **model response** content and choose "Yes" or "No" to answer whether the subsequent **constraint** is met.
## Please provide an objective and fair answer based on the following judgment rules:
- The **constraint** can be understood as a step-by-step scoring point of the **input instruction**, judging whether a specific part is satisfied. Therefore, you only need to consider whether the requirements stated in the **constraint** are met, without focusing on whether the entire **input instruction** is completely satisfied.
Example:
    - **Input instruction**: "Please generate a poem, and end with 'Generation complete.'"
    - **Model response**: "Moonlight before my bed, perhaps frost on the ground. Lifting my head, I gaze at the bright moon, lowering my head, I think of my hometown."
    - **Constraint**: "The poem is written by Li Bai."
For this constraint, although the model response did not complete the **input instruction** requirement to end with "Generation complete," it met the **constraint** requirement, as the poem is Li Bai's "Quiet Night Thoughts," so the answer should be: "Yes." For situations where the **constraint** requirements are met, but the entire **input instruction** is not completely satisfied, the answer should be "Yes."
- Yes: Please check whether the **model response** meets the **constraint**, fully understand the meaning of the **constraint**, don't miss small details, focus only on the current **constraint**, don't pay attention to other requirements in the **input instruction**. The response must perfectly and adequately meet the constraint requirements to be evaluated as "Yes." Even if there is a small error or ambiguous content, it cannot be "Yes." There should be no claims like "basically correct," "mostly correct," or "correct under certain conditions." Such cases should all be evaluated as "No."
- No: If the text in the **model response** cannot meet the requirements of the current **constraint** or does not provide information that can be used to verify the constraint. Choose "No."
Example: If the constraint states "The second sentence of the generated text is a compound sentence" but the **model response** has only one sentence. It does not provide relevant information to verify the constraint. Therefore, the answer should be "No."
## Scoring details:
(1) When evaluating whether the model response is itemized, it must have clear bullet points or numbers to be considered itemized and evaluated as "Yes," otherwise evaluate as "No." Using only connecting words such as "first," "then," "second," "finally," etc., cannot be recognized as itemization and should be evaluated as "No."
(2) When evaluating whether the model response uses a certain language (such as Chinese/English) for output, unless the **input instruction** mentions the need to use multiple languages, it must use only that language to be evaluated as "Yes." If multiple languages are mixed (i.e., words from other languages appear), it should be evaluated as "No."
(3) If a **constraint** contains descriptions like "each," "all," etc., the satisfaction of each object needs to be considered. If any object does not meet the requirements, it should be evaluated as "No"; only when all objects meet the requirements can it be evaluated as "Yes."
(4) For **constraints** stating "The model correctly judges" something, when evaluating whether the model correctly judged a certain selection condition, it is necessary to judge whether the model response correctly chose the corresponding branch based on the instruction requirements of different selection branches in the **input instruction**:
## Output format
Return in json format
```json
{{{{
    "Analysis": "xxx",
    "Answer": Yes/No
}}}}
```
## Evaluation information
**Input instruction**
{input}
**Model response**
{output}
**Constraint**
{constraint}
Please analyze and answer whether the **model response** satisfies the **constraint**:
"""

# Rule evaluation function mapping
RULE_FUNCTIONS = {
    "evaluate_word_length": evaluate_word_length,
    "evaluate_sentence_length": evaluate_sentence_length, 
    "evaluate_keyword": evaluate_keyword,
    "evaluate_format": evaluate_format,
    "evaluate_start_with": evaluate_start_with,
    "evaluate_end_with": evaluate_end_with,
    "check_english_uppercase": check_english_uppercase,
    "check_english_lowercase": check_english_lowercase,
    "contains_no_punctuation": contains_no_punctuation
}

def set_openai_credentials(api_key=None, api_url=None):
    """
    Parameters:
    api_key (str, optional): OpenAI API key, if None, use the existing environment variable
    api_url (str, optional): OpenAI API URL, if None, use the existing environment variable
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_url:
        os.environ["OPENAI_BASE_URL"] = api_url

def call_openai_api(prompt, model="ep-20250214103445-qhtlc", max_retries=3, retry_delay=5):
    """
    Parameters:
        prompt (str): The prompt to be sent to the API
        model (str): The model to be used
        max_retries (int): Maximum number of retries
        retry_delay (int): Delay in seconds between retries

    Returns:
        str: The response from the API
    """
    # 从环境变量获取API密钥和URL
    api_key = os.environ.get("OPENAI_API_KEY")
    api_url = os.environ.get("OPENAI_BASE_URL")
    
    if not api_key:
        raise ValueError("The OpenAI API key was not found. Please set it using set_openai_credentials() or provide it as an environment variable.")
    
    # Configure the client
    client_args = {"api_key": api_key}
    if api_url:
        client_args["base_url"] = api_url
    
    client = OpenAI(**client_args)
    
    # API call with retry mechanism
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0  # Use a low temperature to obtain consistent evaluations
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(f"Error occurred while calling the API: {e}, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.info(f"Reached the maximum number of retries, API call failed: {e}")
                return None

def parse_evaluation_result(raw_response, max_retries=5):
    """
    Parameters:
        raw_response (str): The raw response from GPT-4
        max_retries (int): Maximum number of retries

    Returns:
        dict: A dictionary containing the keys 'Analysis' and 'Answer'
    """
    if not raw_response:
        return {"Analysis": "API call failed", "Answer": "No"}
    
    # 定义解析策略
    strategies = [
        # Strategy 1: Standard JSON code block
        (r'```json\s*(.*?)\s*```', True),
        # Strategy 2: Standard JSON object
        (r'{[\s\S]*?"Analysis"[\s\S]*?:[\s\S]*?".*?"[\s\S]*?,[\s\S]*?"Answer"[\s\S]*?:[\s\S]*?(Yes|No|"Yes"|"No")[\s\S]*?}', False),
        # Strategy 3: Loose JSON object
        (r'{[\s\S]*?"Analysis"[\s\S]*?:[\s\S]*?"[^"]*?"[\s\S]*?,[\s\S]*?"Answer"[\s\S]*?:[\s\S]*?.+?}', False),
        # Strategy 4: Any JSON object
        (r'{.*}', False),
        # Strategy 5: Keyword matching
        (r'\b[Yy][Ee][Ss]\b|\b[Nn][Oo]\b', False)
    ]
    
    for attempt in range(max_retries):
        strategy_idx = min(attempt, len(strategies) - 1)
        pattern, use_group1 = strategies[strategy_idx]
        
        try:
            json_match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            
            # If it is the last strategy (keyword matching)
            if strategy_idx == len(strategies) - 1:
                if json_match:
                    is_yes = json_match.group(0).lower() == "yes"
                    return {
                        "Analysis": f"Extract results using the keyword matching strategy",
                        "Answer": "Yes" if is_yes else "No"
                    }
                else:
                    # default No
                    return {
                        "Analysis": "All parsing attempts failed, defaulting to No",
                        "Answer": "No"
                    }
            
            if json_match:
                json_str = json_match.group(1) if use_group1 else json_match.group(0)
                
                # Attempt to parse JSON
                try:
                    result = json.loads(json_str)
                    
                    # Normalize the answer (case)
                    if isinstance(result.get("Answer"), str):
                        answer = result["Answer"].lower().capitalize()
                        if answer not in ["Yes", "No"]:
                            if answer.strip('"\'').lower() == "yes":
                                answer = "Yes"
                            else:
                                answer = "No"
                        result["Answer"] = answer
                    
                    # Ensure the result contains the required fields
                    if "Analysis" not in result:
                        result["Analysis"] = f"使用策略{strategy_idx+1}提取的结果，无Analysis字段"
                    if "Answer" not in result:
                        result["Answer"] = "No"  # default No
                    
                    return result
                except json.JSONDecodeError as e:
                    continue

            else:
                logger.info(f"Strategy {strategy_idx+1} failed to extract content from the response")
        
        except Exception as e:
            logger.info(f"An exception occurred during parsing (Strategy {strategy_idx+1}): {e}")
        
        # Try the next strategy
        if attempt < max_retries - 1:
            logger.info(f"Attempt {attempt+1} failed, switching to the next strategy...")
    
    # All attempts failed, using the final fallback strategy
    yes_count = len(re.findall(r'\b[Yy][Ee][Ss]\b', raw_response))
    no_count = len(re.findall(r'\b[Nn][Oo]\b', raw_response))
    
    return {
        "Analysis": "All parsing attempts failed, judging based on the number of occurrences of Yes/No keywords",
        "Answer": "Yes" if yes_count > no_count else "No"
    }

def evaluate_constraint_llm(input_prompt, response, constraint, model):
    """
    Parameters:
        input_prompt (str): Input instruction/prompt
        response (str): Model's response
        constraint (str): Constraint to be evaluated
        model (str): Name of the model to be used

    Returns:
        dict: Evaluation result containing the keys 'Analysis' and 'Answer'
    """
    # Format the evaluation prompt
    formatted_prompt = EVALUATION_PROMPT.format(
        input=input_prompt,
        output=response,
        constraint=constraint
    )
    
    # call OpenAI API
    raw_result = call_openai_api(formatted_prompt, model=model)
    
    # parse result
    return parse_evaluation_result(raw_result)

def evaluate_constraint_rule(response, rule_dict):
    """
    Evaluate whether the model's response meets the constraints using rule functions.

    Parameters:
        response (str): Model's response
        rule_dict (dict): Dictionary containing rule functions and parameters

    Returns:
        dict: Evaluation result containing the keys 'Analysis' and 'Answer'
    """
    try:
        func_name = rule_dict.get('func')
        func_input = rule_dict.get('func_input', [])

        # get related function
        if func_name not in RULE_FUNCTIONS:
            return {
                "Analysis": f"Rule function not found: {func_name}",
                "Answer": "No"
            }

        func = RULE_FUNCTIONS[func_name]

        # Construct function arguments, ensuring that the first argument is the response
        if len(func_input) > 0 and func_input[0] != response :
            # Replace the first argument with the current response
            func_args = [response] + func_input[1:]
        else:
            func_args = func_input
        if func_name == "evaluate_format":
            func_args = func_input
        # call function

        result = func(*func_args)
        return {
            "Analysis": f"Evaluate the result using the rule function {func_name}",
            "Answer": "Yes" if result else "No"
        }
    except Exception as e:
        logger.info("")
        logger.info(f"Error occurred during {func_name} rule evaluation: {e}")
        return {
            "Analysis": f"Rule evaluation error: {str(e)}",
            "Answer": "No"
        }

def evaluate_sample(sample, model, constraint_key="added_constraint_from_LLM", response_key="output",
                   enable_llm=True, enable_rule=True, rule_evaluate_key="added_constraint_from_rule"):
    """
    Evaluate all constraints for a single sample, supporting both LLM and rule-based evaluation.
    
    Parameters:
        sample (dict): Sample data
        model (str): Name of the evaluation model
        constraint_key (str): Key name for the list of constraints
        enable_llm (bool): Whether to enable LLM evaluation
        enable_rule (bool): Whether to enable rule-based evaluation
        rule_evaluate_key (str): Key name for the rule evaluation dictionary

    Returns:
        dict: Dictionary containing the evaluation results
    """
    input_prompt = sample.get("input", "")
    output = sample.get(response_key, "")
    constraints = sample.get(constraint_key, [])
    rule_evaluate_dict = sample.get(rule_evaluate_key, {})
    
    if not constraints and not rule_evaluate_dict:
        return {
            "id": sample.get("id", "unknown"),
            "llm_constraints_total": 0,
            "llm_constraints_satisfied": 0,
            "llm_score_average": 0.0,
            "llm_score_binary": 0,
            "rule_constraints_total": 0,
            "rule_constraints_satisfied": 0,
            "rule_score_average": 0.0,
            "rule_score_binary": 0,
            "total_constraints_total": 0,
            "total_constraints_satisfied": 0,
            "total_score_average": 0.0,
            "total_score_binary": 0,
            "llm_constraint_results": [],
            "rule_constraint_results": []
        }
    
    # LLM eval result
    llm_constraint_results = []
    llm_satisfied_count = 0
    
    # rule-based eval result
    rule_constraint_results = []
    rule_satisfied_count = 0
    
    # execute LLM evaluation
    if enable_llm and constraints:
        for i, constraint in enumerate(constraints):
            result = evaluate_constraint_llm(input_prompt, output, constraint, model)
            is_satisfied = result["Answer"] == "Yes"
            
            if is_satisfied:
                llm_satisfied_count += 1
            
            llm_constraint_results.append({
                "constraint_id": i,
                "constraint_text": constraint,
                "is_satisfied": is_satisfied,
                "analysis": result["Analysis"]
            })
    
    # execute rule-based evaluation
    add_constraint_from_rule =  sample["added_constraint_from_rule"]
    if enable_rule and rule_evaluate_dict:
        for rule_type, v_list in add_constraint_from_rule.items():
            for v in v_list:
                if rule_type == "Length":
                    if 'word' in v and 'sentence' not in v:
                        idx = DICTION['Length'][0] # word_length
                    elif 'word' not in v and 'sentence' in v:
                        idx = DICTION['Length'][1] #sentence_length
                else:
                    idx = DICTION[rule_type]   # Convert the type label of the add_constraint_from_rule field to the type label in the rule_evaluate_dict
                rule_info = rule_evaluate_dict[idx]
                if rule_type == "Keyword" and isinstance(rule_info["func_input"], list) and len(rule_info["func_input"]) > 1:
                    # Handle multiple keyword constraints
                    try:
                        keyword_dict = rule_info["func_input"][1]
                        for keyword, num in keyword_dict.items():
                            rule_result = evaluate_constraint_rule(
                                output, 
                                {"func": rule_info["func"], "func_input": [output, keyword, num]}
                            )
                            is_satisfied = rule_result["Answer"] == "Yes"
                            
                            if is_satisfied:
                                rule_satisfied_count += 1
                            
                            rule_constraint_results.append({
                                "constraint_id": f"{rule_type}_{keyword}",
                                "constraint_text": f"The keyword '{keyword}' appears {num} times",
                                "is_satisfied": is_satisfied,
                                "analysis": rule_result["Analysis"]
                            })
                    except Exception as e:
                        logger.info(f"Error occurred while processing keyword constraints: {e}")

                else:

                    rule_result = evaluate_constraint_rule(output, rule_info)
                    is_satisfied = rule_result["Answer"] == "Yes"
                    
                    if is_satisfied:
                        rule_satisfied_count += 1
                    
                    rule_constraint_results.append({
                        "constraint_id": rule_type,
                        "constraint_text": f"rule-based constraint: {rule_type}",
                        "is_satisfied": is_satisfied,
                        "analysis": rule_result["Analysis"]
                    })

    
    # Calculate LLM metrics
    llm_constraints_total = len(constraints) if enable_llm else 0
    llm_score_average = llm_satisfied_count / llm_constraints_total if llm_constraints_total > 0 else 0
    llm_score_binary = 1 if llm_satisfied_count == llm_constraints_total and llm_constraints_total > 0 else 0
    
    # Calculate rule-based metrics
    rule_constraints_total = len(rule_constraint_results) if enable_rule else 0
    rule_score_average = rule_satisfied_count / rule_constraints_total if rule_constraints_total > 0 else 0
    rule_score_binary = 1 if rule_satisfied_count == rule_constraints_total and rule_constraints_total > 0 else 0
    
    # Calculate overall metrics
    total_constraints_total = llm_constraints_total + rule_constraints_total
    total_constraints_satisfied = llm_satisfied_count + rule_satisfied_count
    total_score_average = total_constraints_satisfied / total_constraints_total if total_constraints_total > 0 else 0
    total_score_binary = 1 if llm_score_binary == 1 and rule_score_binary == 1 and total_constraints_total > 0 else 0
    
    return {
        "id": sample.get("id", "unknown"),
        "llm_constraints_total": llm_constraints_total,
        "llm_constraints_satisfied": llm_satisfied_count,
        "llm_score_average": llm_score_average,
        "llm_score_binary": llm_score_binary,
        "rule_constraints_total": rule_constraints_total,
        "rule_constraints_satisfied": rule_satisfied_count,
        "rule_score_average": rule_score_average,
        "rule_score_binary": rule_score_binary,
        "total_constraints_total": total_constraints_total,
        "total_constraints_satisfied": total_constraints_satisfied,
        "total_score_average": total_score_average,
        "total_score_binary": total_score_binary,
        "llm_constraint_results": llm_constraint_results,
        "rule_constraint_results": rule_constraint_results
    }

def process_test_set(test_set_path, output_path, model, constraint_key="added_constraint_from_LLM", response_key="output",
                    rule_evaluate_key="added_constraint_from_rule", enable_llm=True, enable_rule=True,
                    batch_size=5, batch_delay=1, save_detailed=True, num_workers=4):
    """
    Process the test set and evaluate constraint satisfaction, supporting both LLM and rule-based evaluation.
    
    Parameters:
        test_set_path (str): Path to the test set JSON file
        output_path (str): Path to the output CSV file
        model (str): Name of the evaluation model
        constraint_key (str): Key name for the list of constraints
        rule_evaluate_key (str): Key name for the rule evaluation dictionary
        enable_llm (bool): Whether to enable LLM evaluation
        enable_rule (bool): Whether to enable rule-based evaluation
        batch_size (int): Number of samples to process per batch
        batch_delay (int): Delay in seconds between batches
        save_detailed (bool): Whether to save detailed results
        num_workers (int): Number of parallel worker threads
    """
    # load test set
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)[:200]
    
    logger.info(f"Loaded {len(test_set)} test samples")
    
    # Create the output directory for the results (if it does not exist)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare CSV output
    fieldnames = [
        'id', 
        'llm_constraints_total', 'llm_constraints_satisfied', 'llm_score_average', 'llm_score_binary',
        'rule_constraints_total', 'rule_constraints_satisfied', 'rule_score_average', 'rule_score_binary',
        'total_constraints_total', 'total_constraints_satisfied', 'total_score_average', 'total_score_binary'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # If detailed output is enabled, create a detailed results file
    detailed_path = None
    if save_detailed:
        detailed_path = os.path.splitext(output_path)[0] + '_detailed.json'
    
    # Use ThreadPoolExecutor to process samples in parallel
    all_results = []
    
    # Overall statistics
    llm_total_constraints = 0
    llm_total_satisfied = 0
    llm_binary_success_count = 0
    
    rule_total_constraints = 0
    rule_total_satisfied = 0
    rule_binary_success_count = 0
    
    total_constraints = 0
    total_satisfied = 0
    binary_success_count = 0
    
    # Process samples in batches to avoid API rate limits
    batches = [test_set[i:i+batch_size] for i in range(0, len(test_set), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        batch_results = []
        
        # Use a thread pool to process each batch in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_sample = {
                executor.submit(
                    evaluate_sample, 
                    sample, 
                    model, 
                    constraint_key, 
                    response_key,
                    enable_llm, 
                    enable_rule,
                    rule_evaluate_key
                ): sample for sample in batch
            }
            
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    logger.info(f"Error occurred while processing the sample: {e}")
                    # Add the failed sample
                    batch_results.append({
                        "id": sample.get("id", "unknown"),
                        "llm_constraints_total": 0,
                        "llm_constraints_satisfied": 0,
                        "llm_score_average": 0.0,
                        "llm_score_binary": 0,
                        "rule_constraints_total": 0,
                        "rule_constraints_satisfied": 0,
                        "rule_score_average": 0.0,
                        "rule_score_binary": 0,
                        "total_constraints_total": 0,
                        "total_constraints_satisfied": 0,
                        "total_score_average": 0.0,
                        "total_score_binary": 0,
                        "llm_constraint_results": [],
                        "rule_constraint_results": [],
                        "error": str(e)
                    })
        
        # Write the results to CSV
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for result in batch_results:
                # Create a copy without the constraint_results
                row = {k: result[k] for k in fieldnames if k in result}
                writer.writerow(row)
        
        # update statistic
        for result in batch_results:
            # LLM statistics
            llm_total_constraints += result.get("llm_constraints_total", 0)
            llm_total_satisfied += result.get("llm_constraints_satisfied", 0)
            if result.get("llm_score_binary", 0) == 1:
                llm_binary_success_count += 1
            
            # rule-based statistic
            rule_total_constraints += result.get("rule_constraints_total", 0)
            rule_total_satisfied += result.get("rule_constraints_satisfied", 0)
            if result.get("rule_score_binary", 0) == 1:
                rule_binary_success_count += 1
            
            # overall statistic
            total_constraints += result.get("total_constraints_total", 0)
            total_satisfied += result.get("total_constraints_satisfied", 0)
            if result.get("total_score_binary", 0) == 1:
                binary_success_count += 1
        
        # Add the current batch results to the overall results
        all_results.extend(batch_results)
        
        # Add delay between batches (except for the last batch)
        if batch_idx < len(batches) - 1 and batch_delay > 0:
            time.sleep(batch_delay)
    
    # Calculate overall metrics
    llm_avg_score = llm_total_satisfied / llm_total_constraints if llm_total_constraints > 0 else 0
    llm_binary_score = llm_binary_success_count / len(test_set) if test_set else 0
    
    rule_avg_score = rule_total_satisfied / rule_total_constraints if rule_total_constraints > 0 else 0
    rule_binary_score = rule_binary_success_count / len(test_set) if test_set else 0
    
    total_avg_score = total_satisfied / total_constraints if total_constraints > 0 else 0
    total_binary_score = binary_success_count / len(test_set) if test_set else 0
    
    # Print overall metrics
    logger.info("\nevaluation result:")
    logger.info(f"Total number of samples: {len(test_set)}")
    
    if enable_llm:
        logger.info("\nLLM Evaluation Results:")
        logger.info(f"Total number of LLM constraints: {llm_total_constraints}")
        logger.info(f"Number of satisfied LLM constraints: {llm_total_satisfied}")
        logger.info(f"Average satisfaction rate of LLM constraints: {llm_avg_score:.4f}")
        logger.info(f"Proportion of samples with all LLM constraints satisfied: {llm_binary_score:.4f}")
    
    if enable_rule:
        logger.info("\nRule Evaluation Results:")
        logger.info(f"Total number of rule constraints: {rule_total_constraints}")
        logger.info(f"Number of satisfied rule constraints: {rule_total_satisfied}")
        logger.info(f"Average satisfaction rate of rule constraints: {rule_avg_score:.4f}")
        logger.info(f"Proportion of samples with all rule constraints satisfied: {rule_binary_score:.4f}")

    logger.info("\nComprehensive Evaluation Results:")
    logger.info(f"Total number of constraints: {total_constraints}")
    logger.info(f"Number of satisfied constraints: {total_satisfied}")
    logger.info(f"Average satisfaction rate of constraints: {total_avg_score:.4f}")
    logger.info(f"Proportion of samples with all constraints satisfied: {total_binary_score:.4f}")
    
    # Save detailed results (if enabled)
    if detailed_path:
        with open(detailed_path, 'w', encoding='utf-8') as f:
            detailed_results = {
                "summary": {
                    "total_samples": len(test_set),
                    "llm_evaluation": {
                        "enabled": enable_llm,
                        "total_constraints": llm_total_constraints,
                        "total_satisfied": llm_total_satisfied,
                        "average_score": llm_avg_score,
                        "binary_score": llm_binary_score
                    },
                    "rule_evaluation": {
                        "enabled": enable_rule,
                        "total_constraints": rule_total_constraints,
                        "total_satisfied": rule_total_satisfied,
                        "average_score": rule_avg_score,
                        "binary_score": rule_binary_score
                    },
                    "combined_evaluation": {
                        "total_constraints": total_constraints,
                        "total_satisfied": total_satisfied,
                        "average_score": total_avg_score,
                        "binary_score": total_binary_score
                    }
                },
                "results": all_results
            }
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results have been saved to {detailed_path}")
    
    logger.info(f"The evaluation results have been saved to {output_path}")
    
    return {
        "llm_average_score": llm_avg_score,
        "llm_binary_score": llm_binary_score,
        "rule_average_score": rule_avg_score,
        "rule_binary_score": rule_binary_score,
        "total_average_score": total_avg_score,
        "total_binary_score": total_binary_score
    }

def main():
    """Main function, handle command-line arguments and run evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate the constraint satisfaction of model responses in the test set")
    
    # Required parameters
    parser.add_argument("--test_set", default="../test_set/test_number_10.json", help="Path to the test set JSON file")
    parser.add_argument("--output",
                        default="../evaluation_example/RECAST_Test_results/data/test1",
                        help="Path to the output CSV file")
    # API options
    parser.add_argument("--api_key",default="your_api_key", help="OpenAI API key")
    parser.add_argument("--api_url",default='your_api_url', help="OpenAI API base URL")
    parser.add_argument("--model", default="qwen-plus", help="Name of the evaluation model to be used")
    
    # Constraint options
    parser.add_argument("--constraint_key", default="added_constraint_from_LLM",
                        help="Key name for the list of constraints")
    parser.add_argument("--rule_evaluate_key", default="rule_evaluate_dict",
                        help="Key name for the rule evaluation dictionary")
    parser.add_argument("--response_key", default="response",
                        help="Key name for the response in the rule evaluation dictionary")
    # Evaluation method options
    parser.add_argument("--enable_llm", action="store_true", default=True,
                        help="Enable LLM evaluation")
    parser.add_argument("--enable_rule", action="store_true", default=True,
                        help="Enable rule evaluation")
    parser.add_argument("--disable_llm", action="store_true", default=False,
                        help="Disable LLM evaluation")
    parser.add_argument("--disable_rule", action="store_true", default=False,
                        help="Disable rule evaluation")
    
    # Batch processing options
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of samples to process per batch")
    parser.add_argument("--batch_delay", type=int, default=1,
                        help="Delay in seconds between batches")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of parallel worker threads")

    # Other options
    parser.add_argument("--no_detailed", action="store_true",
                        help="Disable detailed result output")
    
    args = parser.parse_args()
    
    # Process evaluation method options
    enable_llm = args.enable_llm and not args.disable_llm
    enable_rule = args.enable_rule and not args.disable_rule
    
    # Ensure at least one evaluation method is enabled
    if not enable_llm and not enable_rule:
        logger.error("Error: Both LLM evaluation and rule evaluation are disabled. At least one evaluation method must be enabled.")
        return
    
    # Set API credentials
    set_openai_credentials(args.api_key, args.api_url)
    
    # Process the test set
    process_test_set(
        test_set_path=args.test_set,
        output_path=args.output,
        model=args.model,
        constraint_key=args.constraint_key,
        response_key = args.response_key,
        rule_evaluate_key=args.rule_evaluate_key,
        enable_llm=enable_llm,
        enable_rule=enable_rule,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
        save_detailed=not args.no_detailed,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
