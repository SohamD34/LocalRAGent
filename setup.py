import subprocess
import time
from dotenv import load_dotenv
import os
load_dotenv()



def wait():
    '''
        Wait for user input to proceed
    '''
    print('.')
    time.sleep(1)



def install_dependencies():
    ''' 
        Install all the dependencies required for the project
    '''
    print('Installing dependencies...')

    try:
        subprocess.run('pip install -q -U langchain-nomic langchain_community langchainhub langchain', shell=True)
        wait()
        subprocess.run('pip install -q -U tiktoken chromadb', shell=True)
        wait()
        subprocess.run('pip install -q -U langgraph tavily-python gpt4all', shell=True)
        wait()
        subprocess.run('pip install -q -U firecrawl-py', shell=True)
        wait()

    except Exception as e:
        print('Error installing dependencies: ', e)
        return

    print("Dependencies installed successfully !!")



def freeze_dependencies():
    '''
        Freeze all the dependencies to requirements.txt file
    '''
    try:
        subprocess.run('pip freeze > requirements.txt', shell=True)
    except:
        raise Exception('Error freezing dependencies')
    



def create_env_file():
    '''
        Create .env file with the API keys, endpoints, environment variables and other configurations
    '''
    langchain_api = input('Enter your langchain API key: ')

    if len(langchain_api) > 1:
        pass
    else:
        print('Using default API key...')
        langchain_api = os.getenv('LANGCHAIN_API_KEY')
        wait()
        
    try:
        with open('.env', 'w') as f:
            f.write(f"LANGCHAIN_API_KEY = '{langchain_api}'")   
            f.write(f"\nLANGCHAIN_ENDPOINT = 'https://api.smith.langchain.com'")
            f.write(f"\nLANGCHAIN_TRACING_V2 = 'true'")
            f.write(f"\nLANGCHAIN_HUB_API_URL = 'https://api.smith.langchain.com'")
    except:
        raise Exception('\nError creating .env file')

    wait()
    print('Environment created !!')
    wait()


if __name__ == '__main__':
    install_dependencies()
    wait()
    freeze_dependencies()
    wait()
    create_env_file()
    wait()
    print('Setup completed successfully !! Your environment is now ready to use and build !!\n\n')