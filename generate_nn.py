""" Main script used to generate NN dataset using GPT-5"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import config
import utils
from verify_nn import check_all_requirements



def generate_nn_and_save(task: str, input_: str, architecture: str,
                         complexity: str, file_path: str):
    """
    Generates NN code based on the given specifications
    and saves it to a file.
    
    Parameters:
        task (str): Task requirement.
        input_ (str): Input requirement.
        architecture (str): Type of architecture requirement.
        complexity (str): Complexity requirement.
        file_path (str): Path to the file where the generated 
            code will be saved.
    
    Returns:
        None
    """
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt = config.PROMPT_TEMPLATE.format(
        architecture=architecture, task=task,
        input_=input_, complexity=complexity
    )

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )
    generated_code = response.choices[0].message.content
    cleaned_code = generated_code.replace("```python", "").replace("```", "")
    cleaned_code = cleaned_code.strip()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"""'''{prompt}'''\n""")
        f.write(cleaned_code)

    print(f"Code saved to {file_path}")



def main():
    """
    Generates the NN code and verifies compliance with the requirements.
    """
    inputs = utils.construct_input_statements(config.inputs_dict)
    archi = config.architectures[0]

    for input_ in inputs[:2]:
        for task in config.tasks[0:3]:
            for cmpl in config.complexities[0:3]:
                complexity_summary = utils.generate_complexity_def(
                    archi, input_
                )
                complexity = f"{cmpl}: {complexity_summary[cmpl]}"
                input_parsed = utils.parse_input_for_file_name(input_)
                kwds = [archi, task, input_parsed, cmpl]
                file_name = "_".join(k.lower() for k in kwds) + ".py"
                file_path = f"dataset_nns/{file_name}"
                generate_nn_and_save(
                    task, input_, archi, complexity, file_path
                )
                check_all_requirements(
                    task, input_, archi, complexity, file_path
                )

if __name__ == "__main__":
    main()
