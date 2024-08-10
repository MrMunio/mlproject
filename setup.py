from setuptools import find_packages, setup

def get_requirements(file_path:str)->list[str]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.replace("\n","") for line in lines] 
        lines = [line.strip() for line in lines]
        if "-e ." in lines:
            lines.remove("-e .")
        return lines

setup(
    name="ML_Project_generic",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    author="muni sekhar boligala",
    author_email="sekharmuni003@gmail.com",
    description="A simple and generic machine learning project"
)