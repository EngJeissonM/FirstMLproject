from setuptools import find_packages,setup

from typing import List #this is to import the List type from the typing module

#this will install the requirements from the requirements.txt file
#this is a hyphen-e-dot variable to ignore the -e . in the requirements.txt
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:

    requirements=[]
    with open(file_path) as file_obj: #this will open the file in read mode
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

#this creates a package adn version, and call the function to install the requirements
setup(
name='MLproject',
version='0.0.1',
author='EngJeissonM',
author_email='jsmoraleshengineer@gmail.com',
packages=find_packages(), #this will find all the packages in the project
install_requires=get_requirements('requirements.txt')

)