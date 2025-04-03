import os
import sys
sys.path.append("./apis")
import time
import numpy as np
import re
import ast
import signal
import importlib.util
import json
import tiktoken
from openai import AzureOpenAI
import traceback
import argparse
import gurobipy as gp
import env_0
from apis.z_descriptions import description_d0, get_code_lines, get_code_usage
import utils

def parse_direct_result(response, value_type="array"):
    try:
        print("*"*32)
        print("*"*32)
        print("*"*32)
        print()
        print("RESPONSE is", response)
        print()
        print("*"*32)
        print("*"*32)
        print("*"*32)
        match = re.search(r"ANSWER\s*=\s*(\[[^\]]*\])", response)
        if match:
            result = ast.literal_eval(match.group(1))
            if value_type=="array":
                return np.array(result)
            elif value_type=="list":
                return [xxx for xxx in result]
            else:
                return result
        else:
            return None
    except:
        return None

def parse_to_code(response, filepath):
    try:
        code_block_pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
        code_blocks = code_block_pattern.findall(response)
        if code_blocks:
            with open(filepath, "w") as f:
                for item in code_blocks:
                    if "import numpy as np" not in item:
                        f.write("import numpy as np\n")
                    if "import casadi as ca" not in item:
                        f.write("import casadi as ca\n")
                    f.write(item)
                    break
        return True
    except:
        return False

def load_function_from_file(filepath, func_name):
    try:
        error_info = None
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # register module in sys.modules
        spec.loader.exec_module(module)
        llm_function = getattr(module, func_name)
        return llm_function, error_info
    except (ImportError, AttributeError, FileNotFoundError, SyntaxError) as e:
        error_info = f"Failed to load '{func_name}' from '{filepath}': {e}"
        error_obj = traceback.print_exc()
        if error_obj is not None:
            error_info += str(error_obj)
        return None, error_info
    except Exception as e:
        print(f"Unexpected error occurred while loading the module: {e}")
        error_obj = traceback.print_exc()
        if error_obj is not None:
            error_info += str(error_obj)
        return None, error_info
    except:
        print("UNKNOWN ERROR caused exception")
        error_obj = traceback.print_exc()
        if error_obj is not None:
            error_info += str(error_obj)
        return None, error_info


def handle_LLM_solution(llm_mode, response, env, timeout_thres, round, trial_i):
    ### Parsing output ###
    is_parsing_error = False
    is_loading_error = False
    is_runtime_error = False
    is_timeout = False
    load_error_trace = None
    runtime_error_trace = None
    
    if llm_mode=="END2END":
        sols = parse_direct_result(response)
        if sols is None:
            is_parsing_error = True
    elif llm_mode in ["CODE", "API"]:
        filepath = "%s/trial_%04d_r%d_llm_code.py"%(args.exp_dir_full, trial_i, round)
        is_valid_code_status = parse_to_code(response, filepath)
        if is_valid_code_status:
            llm_function, error_info = load_function_from_file(filepath, func_name="find_path")
            if llm_function is not None:
                try:
                    try:
                        signal.setitimer(signal.ITIMER_REAL, timeout_thres)
                        sols = llm_function(env)
                    except TimeoutError as e:
                        is_timeout = True
                        sols = None
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                except:
                    runtime_error_trace = traceback.format_exc()                   
                    # Case 1: Runtime error while executing the function
                    print(f"Error executing LLM solution:\n{runtime_error_trace}")
                    sols = None
                    is_runtime_error = True
            else:
                # Case 2: Function loading failed (e.g., syntax error or missing function)
                sols = None
                is_loading_error = True
                load_error_trace = error_info
        else:
            # Case 3: Code parsing failed (e.g., couldn't extract code from LLM response)
            sols = None
            is_parsing_error = True
    return sols, is_parsing_error, is_loading_error, is_runtime_error, is_timeout, load_error_trace, runtime_error_trace


def sanitize_messages(messages):
    cleaned = []
    for i, m in enumerate(messages):
        if not isinstance(m.get("content"), str) or m["content"].strip() == "":
            print(f"[Warning] Dropping invalid message[{i}] with role={m.get('role')} and content={m.get('content')}")
            continue
        cleaned.append(m)
    return cleaned

def call_llm_api(client, tmp_prompt, seed, temperature, tmp_messages=None):
    if tmp_messages is None:
        if args.model_name!="o1-mini":
            tmp_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        else:
            tmp_messages = []
    tmp_messages.append({"role": "user", "content": tmp_prompt})
    tmp_messages = sanitize_messages(tmp_messages)
    if args.model_name in ["o1", "o1-mini", "o3-mini"]:
        tmp_completion = client.chat.completions.create(
            model=args.model_name,
            seed=seed,
            max_completion_tokens=args.max_tokens,
            messages=tmp_messages,
        )
    else:
        tmp_completion = client.chat.completions.create(
            model=args.model_name,
            seed=seed,
            temperature=temperature,
            max_tokens=args.max_tokens,
            messages=tmp_messages,
        )
    
    tmp_response = tmp_completion.choices[0].message.content
    tmp_messages.append({"role": "assistant", "content": tmp_response})
    return tmp_response, tmp_messages

def write_flag_line(s):
    n_width = max(32, len(s) + 8)
    n_rest = (n_width - len(s))//2
    s1 = "%s\n"%("#"*n_width)
    s2 = "%s%s%s%s%s\n"%("#"*2, " "*(n_rest-2), " "*len(s), " "*(n_width-2-n_rest-len(s)), "#"*2)
    s3 = "%s%s%s%s%s\n"%("#"*2, " "*(n_rest-2), s, " "*(n_width-2-n_rest-len(s)), "#"*2)
    s4 = "%s%s%s%s%s\n"%("#"*2, " "*(n_rest-2), " "*len(s), " "*(n_width-2-n_rest-len(s)), "#"*2)
    s5 = "%s\n"%("#"*n_width)
    return s1 + s2 + s3 + s4 + s5

def write_message_log(round=0, type="prompt", message=None, trial_i=0):
    if message is not None:
        if type!="code":
            with open("%s/trial_%04d_r_conversation.txt"%(args.exp_dir_full, trial_i), "a") as f:
                f.write(write_flag_line("Round-%d: %s"%(round if round!=-1 else 0.5, type)))
                f.write(message+"\n"+"\n")
        if type in ["prompt", "response"]:
            if round==-1:
                with open("%s/trial_%04d_r0_%s.txt"%(args.exp_dir_full, trial_i, type), "a") as f:
                    f.write(message)
            else:
                with open("%s/trial_%04d_r%d_%s.txt"%(args.exp_dir_full, trial_i, round, type), "w") as f:
                    f.write(message)
        elif type=="code":
            with open("%s/trial_%04d_r%d_z%s.py"%(args.exp_dir_full, trial_i, round, type), "w") as f:
                f.write(message)

def main():
    utils.setup_exp_and_logger(args)
    
    # LLM configs
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    default_api_str_list = "astar, cem, grad, lqr, milp, mpc, pid, rrt".split(", ")
    
    # initialize LLM API
    client = AzureOpenAI(
        api_version=args.api_version,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    
    eta = utils.EtaEstimator(0, args.num_trials)
    
    # Set timeout alarm
    def timeout_handler(signum, frame):
        raise TimeoutError("Function execution exceeded %.1f seconds"%(args.timeout_thres))
    signal.signal(signal.SIGALRM, timeout_handler)
    
    met_d_list_dict={}
    for trial_i in range(args.num_trials):
        eta.update()
        X_MIN, X_MAX, Y_MIN, Y_MAX = 0., 10., 0., 10.

        env_seed = args.env_seed + (trial_i // args.n_dup_runs)
        if args.task_type in [4, 5]:  # STL
            # initialize env
            env = env_0.ControlEnv(
                dynamics_type=args.dynamics_type,
                task_type=args.task_type,
                seed=env_seed,
                x_min=np.array([X_MIN, Y_MIN]),
                x_max=np.array([X_MAX, Y_MAX]),
                u_min=np.array([-4., -4.]),
                u_max=np.array([4., 4.]),
                num_obstacles=12,
                obstacle_type="mixed",
                min_clearance=0.5,
                min_radius=0.8,
                max_radius=2.0,
                dt=1.0,
                nt=20
            )
        elif args.task_type==3: # HIER
            # initialize env
            env = env_0.ControlEnv(
                dynamics_type=args.dynamics_type,
                task_type=args.task_type,
                seed=env_seed,
                x_min=np.array([X_MIN, Y_MIN]),
                x_max=np.array([X_MAX, Y_MAX]),
                u_min=np.array([-1., -1.]),
                u_max=np.array([1., 1.]),
                num_obstacles=4,
                obstacle_type="mixed",
                min_clearance=0.5,
                min_radius=0.8,
                max_radius=2.0,
                dt=1.0,
                nt=20
            )
        elif args.task_type==0: # tracking
            if args.dynamics_type=="unicycle":
                env = env_0.ControlEnv(
                    dynamics_type=args.dynamics_type,
                    task_type=args.task_type,
                    seed=env_seed,
                    x_min=np.array([X_MIN, Y_MIN]),
                    x_max=np.array([X_MAX, Y_MAX]),
                    u_min=np.array([-.3, -.5]),
                    u_max=np.array([.3, .5]),
                    num_obstacles=12,
                    obstacle_type="mixed",
                    min_clearance=0.5,
                    min_radius=0.8,
                    max_radius=2.0,
                    dt=1.0,
                    nt=20
                )
            else:
                env = env_0.ControlEnv(
                    dynamics_type=args.dynamics_type,
                    task_type=args.task_type,
                    seed=env_seed,
                    x_min=np.array([X_MIN, Y_MIN]),
                    x_max=np.array([X_MAX, Y_MAX]),
                    u_min=np.array([-.5, -.5]),
                    u_max=np.array([.5, .5]),
                    num_obstacles=12,
                    obstacle_type="mixed",
                    min_clearance=0.5,
                    min_radius=0.8,
                    max_radius=2.0,
                    dt=1.0,
                    nt=20
                )
        else:
            # initialize env
            env = env_0.ControlEnv(
                dynamics_type=args.dynamics_type,
                task_type=args.task_type,
                seed=env_seed,
                x_min=np.array([X_MIN, Y_MIN]),
                x_max=np.array([X_MAX, Y_MAX]),
                u_min=np.array([-.5, -.5]),
                u_max=np.array([.5, .5]),
                num_obstacles=12,
                obstacle_type="mixed",
                min_clearance=0.5,
                min_radius=0.8,
                max_radius=2.0,
                dt=1.0,
                nt=20
            )
        
        meta_d = {"env_seed":env_seed, "task_type":args.task_type, "trial_i":trial_i,}
        met_d_list=[{"meta":meta_d}]
    
        prompt="Imagine you are an expert in planning and control for robotics. "
        prompt+=env.description+"\n"
        env_api_description = env.get_env_api_description()
    
        if args.llm_mode=="END2END":
            if args.task_type in [0, 2, 5]:  # tracking -> controls
                prompt+="Please directly generate the low-level control solution in the format `ANSWER=[(u0, v0), (u1, v1), ..., (u{n-1}, v{n-1})]` in a ```plaintxt block```. "
            elif args.task_type==[1, 3]:  # planning  -> waypoints
                prompt+="Please directly generate the waypoints solution in the format `ANSWER=[(x0, y0), (x1, y1), ..., (xn, yn)]` in a ```plaintxt block```. "
            else:  # time planning -> timed waypoints
                prompt+="Please directly generate the timed waypoints solution in the format `ANSWER=[(x0, y0, t0), (x1, y1, t1), ..., (xn, yn, tn)] in a ```plaintxt block```. "
            prompt+="Be mindful of the potential computation time (hard constraint: %.1f seconds for execution or failure). "%(args.timeout_thres)
        elif args.llm_mode=="CODE":
            prompt+="Please implement a python function `find_path(env)` to generate the solution. "
            prompt+="The env object has APIs that you can call: %s "%(env_api_description)
            prompt+="Must include all obstacles and constraints inside the `find_path(env)`. This function shouldn't call any other parameters. Don't show any example usage."
            prompt+="Try to keep this code simple - don't write any test function - we just need `find_path(env)`.\n"
            prompt+="Be mindful of the potential computation time (hard constraint: %.1f seconds for execution or failure). "%(args.timeout_thres)
            prompt+="Try to make the algorithm as computation-efficient as possible. "
        else:
            the_description_d = description_d0    
            tool_api_description = "You can use `from apis.xxx import solve_sequence as solve_sequence_xxx` where xxx is in {astar, cem, grad, lqr, milp, mpc, pid, rrt}\n"+\
            "- ASTAR: %s\n"%(the_description_d["astar"])+\
            "- CEM: %s\n"%(the_description_d["cem"])+\
            "- GRAD: %s\n"%(the_description_d["grad"])+\
            "- LQR: %s\n"%(the_description_d["lqr"])+\
            "- MILP: %s\n"%(the_description_d["milp"])+\
            "- MPC: %s\n"%(the_description_d["mpc"])+\
            "- PID: %s\n"%(the_description_d["pid"])+\
            "- RRT: %s\n"%(the_description_d["rrt"])
            
            prompt+="We provide you some env APIs and tool APIs you can call. "
            prompt+="The env APIs you can call are: %s "%(env_api_description)
            prompt+="The tool APIs are for planning and control methods. "
            prompt+="%s "%(tool_api_description)
            
            prompt+="Please implement a python function `find_path(env)` to generate the solution via these APIs. "
            prompt+="Be mindful of the potential computation time (hard constraint: %.1f seconds for execution or failure). "%(args.timeout_thres)
            prompt+="Try to use 1 APIs for path-planning problems, and 1-3 APIs for more complex problems. "
            prompt+="The output solution should be a 2D numpy array with the first dimension the path/control/horizon length. "
            if not args.all_apis:
                prompt+="Now a quick pause here -  only list of the tool API name(s) you need to use (1~3 APIs), in the format of `ANSWER=['foo',...]` in a ```plaintxt block```"
        
        if not args.all_apis:
            write_message_log(0, "prompt", prompt, trial_i=trial_i)
            print("%s\n%s\n%s"%("#"*30, prompt, "#"*30))
        
            tokens = tiktoken.encoding_for_model("gpt-4").encode(prompt)
            print(f"Number of tokens: {len(tokens)}")

            ############# Send prompt to LLM to generate output  ################
            response, messages = call_llm_api(client, prompt, seed=args.seed*10000+trial_i*10, temperature=args.temperature)
            
            write_message_log(0, "response", response, trial_i=trial_i)
            print(response)
    
        #####################################################################
        if args.llm_mode=="API":
            if not args.all_apis:
                api_strs = parse_direct_result(response, value_type="list")
                prompt = "Here is the code implementation of the apis:%s. "%(api_strs)
                for api_str in api_strs:
                    if api_str.lower() in default_api_str_list:
                        prompt += "Here is the code for '%s'\n"%(api_str)
                        prompt += "```python\n"
                        if api_str=="milp":
                            prompt += get_code_usage("apis/"+api_str.lower()+".py") + "\n"
                        else:
                            prompt += get_code_lines("apis/"+api_str.lower()+".py") + "\n"
                        prompt += "```\n"
                    else:
                        prompt += "The code for '%s' is NOT FOUND.\n"%(api_str)
                prompt+="\nNow you should understand better about these APIs. Discuss your highlevel plan for using the APIs here. "
                prompt+="If you plan to use multiple APIs, first explain the I/O relationship between these APIs and ensure their shape and type are matched. "
                prompt+="And if you feel you can directly code to get the final result, you can also ignore to use any tool APIs. "
                prompt+="Import necessary libraries and selected APIs (`from xxx import solve_sequence as solve_sequence_xxx`) and their defined classes if any (e.g. Node), and generate the code to for `find_path(env)`. "
                response, messages = call_llm_api(client, prompt, seed=args.seed*10000+trial_i*10, temperature=args.temperature, tmp_messages=messages)
                
                write_message_log(-1, "prompt", prompt, trial_i=trial_i)
                write_message_log(-1, "response", response, trial_i=trial_i)
                met_d_list[0]["apis"] = api_strs
            else:
                prompt += "\nHere is the code implementation of the apis:%s. "%(default_api_str_list)
                for api_str in default_api_str_list:
                    if api_str.lower() in default_api_str_list:
                        prompt += "Here is the code for '%s'\n"%(api_str)
                        prompt += "```python\n"
                        if api_str=="milp":
                            prompt += get_code_usage("apis/"+api_str.lower()+".py") + "\n"
                        else:
                            prompt += get_code_lines("apis/"+api_str.lower()+".py") + "\n"
                        prompt += "```\n"
                    else:
                        prompt += "The code for '%s' is NOT FOUND.\n"%(api_str)
                
                prompt+="\nNow you should understand better about these APIs. Discuss your highlevel plan for using the APIs here. "
                prompt+="If you plan to use multiple APIs, first explain the I/O relationship between these APIs and ensure their shape and type are matched. "
                prompt+="And if you feel you can directly code to get the final result, you can also ignore to use any tool APIs. "
                prompt+="Import necessary libraries and selected APIs (`from xxx import solve_sequence as solve_sequence_xxx`) and their defined classes if any (e.g. Node), and generate the code to for `find_path(env)`. "
            
                tokens = tiktoken.encoding_for_model("gpt-4").encode(prompt)
                print(f"Number of tokens: {len(tokens)}")
                write_message_log(0, "prompt", prompt, trial_i=trial_i)
                
                response, messages = call_llm_api(client, prompt, seed=args.seed*10000+trial_i*10, temperature=args.temperature)
                
                write_message_log(0, "response", response, trial_i=trial_i)
                met_d_list[0]["apis"] = default_api_str_list
    
        ### Parsing output ###
        sols, is_parsing_error, is_loading_error, is_runtime_error, is_timeout, load_error_trace, runtime_error_trace =\
            handle_LLM_solution(args.llm_mode, response, env, args.timeout_thres, round=0, trial_i=trial_i)
        
        eval_result = env.evaluate(sols)
        
        for feedback_i in range(args.num_feedbacks):
            print("FEEDBACK",feedback_i,"$"*25)
            met_d_list[feedback_i]["tracking_error"] = eval_result["tracking_error"]
            if "success" in eval_result and eval_result["success"]:
                met_d_list[feedback_i]["status"] = "success"
                break
            else:
                prompt = ""
                prompt += "This is round-%d. The solution did not work. Here is the diagnose.\n"%(feedback_i+1)
                if is_parsing_error:
                    prompt += "PARSING ERROR: cannot parse the LLM output to meaningful %s.\n"%("value" if args.llm_mode=="END2END" else "code")
                    met_d_list[feedback_i]["status"] = "parsing error"
                elif is_loading_error:
                    prompt += f"SYNTAX ERROR: {load_error_trace}.\n"
                    met_d_list[feedback_i]["status"] = "syntax error"
                elif is_runtime_error:
                    prompt += f"RUNTIME ERROR: {runtime_error_trace}.\n"
                    met_d_list[feedback_i]["status"] = "runtime error"
                elif is_timeout:
                    prompt += "TIMEOUT ERROR: The algorithm ran out of time budget %.1f seconds, please consider more efficient algorithm implementation.\n"%(args.timeout_thres)
                    met_d_list[feedback_i]["status"] = "timeout error"
                else:
                    prompt += eval_result["diagnose"]
                    met_d_list[feedback_i]["status"] = "failure"
                prompt += "Can you re-examine and update your solution?"
                prompt += "Keep the solution in the original format specified in round-0 prompt."
                response, messages = call_llm_api(client, prompt, seed=args.seed*10000+trial_i*10+feedback_i+1, temperature=args.temperature, tmp_messages=messages)
                
                print("*" * 25 + "FEEDBACK", feedback_i, "*" * 25)
                print("*" * 25 + "PROMPT" + "*" * 25)
                print(prompt)
                print("*" * 25 + "RESPONSE" + "*" * 25)
                print(response)
                
                write_message_log(feedback_i+1, "prompt", prompt, trial_i=trial_i)
                write_message_log(feedback_i+1, "response", response, trial_i=trial_i)
            
                sols, is_parsing_error, is_loading_error, is_runtime_error, is_timeout, load_error_trace, runtime_error_trace \
                    = handle_LLM_solution(args.llm_mode, response, env, args.timeout_thres, round=feedback_i+1, trial_i=trial_i)
                eval_result = env.evaluate(sols)
                met_d_list.append({})

        signal.setitimer(signal.ITIMER_REAL, 0)
        if "trajs" in eval_result and eval_result["trajs"] is not None:
            trajs = eval_result["trajs"]
        else:
            trajs = None

        met_d_list[-1]["tracking_error"] = eval_result["tracking_error"]
        if "success" in eval_result and eval_result["success"]:
            met_d_list[-1]["status"] = "success"
        elif is_parsing_error:
            met_d_list[-1]["status"] = "parsing error"
        elif is_loading_error:
            met_d_list[-1]["status"] = "syntax error"
        elif is_runtime_error:
            met_d_list[-1]["status"] = "runtime error"
        elif is_timeout:
            met_d_list[-1]["status"] = "timeout error"
        else:
            met_d_list[-1]["status"] = "failure"
        
        if trajs is not None:
            met_d_list[-1]["trajs"] = trajs.tolist()
        else:
            met_d_list[-1]["trajs"] = None
        
        with open("%s/trial_%04d_metrics.json"%(args.exp_dir_full, trial_i), "w") as f:
            json.dump(met_d_list, f)

        met_d_list_dict[trial_i] = met_d_list
        with open("%s/all_metrics.json"%(args.exp_dir_full), "w") as f:
            json.dump(met_d_list_dict, f)

        # visualization
        env.visualization(trajs)
        utils.plt_save_close("%s/trial_%04d_viz.png"%(args.viz_dir, trial_i))
        
        # print log
        n_success = np.sum([tmp_met_d_list[-1]["status"]=="success" for _, tmp_met_d_list in met_d_list_dict.items()])
        n_rounds = np.sum([len(tmp_met_d_list) for _, tmp_met_d_list in met_d_list_dict.items()])
        print_str="#TRIAL[%04d/%04d] success:%.2f  rounds:%.2f | time/trial:%s  elapsed:%s  ETA:%s"%(
            trial_i, args.num_trials, n_success/(trial_i+1), n_rounds/(trial_i+1),
            eta.interval_str(), eta.elapsed_str(), eta.eta_str()
        )
        print(print_str)
        with open("%s/epoch_logs.txt"%(args.exp_dir_full), "a") as f:
            f.write(print_str+"\n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("AUDERE: Automatic Strategy Design and Realization for Robot Planning and Control via LLMs")
    add = parser.add_argument
    add("--seed", type=int, default=1007)
    add("--env_seed", type=int, default=1007)
    add("--temperature", '-tau', type=float, default=0.1)
    add("--exp_name", '-e', type=str, default="llm_DEBUG")
    add("--gpus", type=str, default="0")
    add("--cpu", action='store_true', default=False)
    add("--test", action='store_true', default=False)
    add("--api_version", type=str, default="2024-02-01")
    add("--model_name", '-M', type=str, default="gpt-4o", choices=["gpt-4o", "gpt-4o-full", "gpt-35-turbo-16k", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"])
    add("--max_tokens", type=int, default=8192)
    add("--dynamics_type", '-D', type=str, default="single", choices=["single", "double", "unicycle", "drones"])
    add("--timeout_thres", type=float, default=10.0)
    add("--num_feedbacks", type=int, default=5)
    add("--llm_mode", type=str, default="API", choices=["END2END", "API", "CODE"])
    add("--task_type", type=int, default=0)
    add("--num_trials", type=int, default=100)
    add("--n_dup_runs", type=int, default=5)
    add("--lite", action='store_true', default=False)
    add("--local", '-L', action='store_true', default=False)
    add("--all_apis", action='store_true', default=False)
        
    args = parser.parse_args()
    args.gpus = None
    args.cpu = True
    args.local = True
    
    if args.lite:
        args.exp_name += "_lite"
        args.num_trials = 10
        args.n_dup_runs = 2
        
    if args.model_name in ["o1", "o3-mini"]:
        args.api_version="2024-12-01-preview"
    
    try:
        gp.setParam("TimeLimit", args.timeout_thres) 
        t1=time.time()
        main()
    except Exception as e:
        print("Program crashes")
        print(f"Exception type: {type(e).__name__}")
        print(f"Message: {e}")
        traceback.print_exc()
    finally:
        t2=time.time()
        print("Finished in %.4f seconds"%(t2-t1))