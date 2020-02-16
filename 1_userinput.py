
import params
import pandas as pd

# load baby names sourced from https://www.ssa.gov/oact/babynames/limits.html

names = pd.read_csv('yob2018.txt', sep=",", header=None)
names.columns = ['name', 'sex', 'baby_count']
names['weight'] = round(names['baby_count']/10)+1

# split names into two files by sex

girl_names = names[names['sex'] == 'F']
boy_names = names[names['sex'] == 'M']

girl_names_sample = girl_names['name'].sample(n = 500, random_state = 2, weights = girl_names['weight'])
boy_names_sample = boy_names['name'].sample(n = 500, random_state = 2, weights = boy_names['weight'])

# request user feedback on baby names by sex

columns = ['name','response']

girl_name_feedback = pd.DataFrame(columns=columns)
boy_name_feedback = pd.DataFrame(columns=columns)

question = "Do you like the name "

def name_feedback():
    """ """
    print("To start, I need to get a basic understanding of the baby names you like.")
    print("This will involve getting feedback on 500 girl names and 500 boy names.")
    print("For each name, respond 'y' if you like the name and 'n' if you don't.")
    global girl_name_feedback
    global boy_name_feedback
    for n in girl_names_sample:
        usr_response = input(question + n + " (y/n)?:").lower().strip()
        print("")
        while not (usr_response == "y" or usr_response == "n"):
            print("Please input y or n")
            usr_response = input(question + n + "(y/n)?:").lower().strip()
            print("")
        a = {
            'name': n,  # some formula for obtaining values
            'response':usr_response
            }
        girl_name_feedback = girl_name_feedback.append(a,ignore_index = True)
    for n in boy_names_sample:
        usr_response = input(question + n + " (y/n):").lower().strip()
        print("")
        while not (usr_response == "y" or usr_response == "n"):
            print("Please input y or n")
            usr_response = input(question + n + "(y/n)?:").lower().strip()
            print("")
        a = {
            'name': n,  # some formula for obtaining values
            'response':usr_response
            }
        boy_name_feedback = boy_name_feedback.append(a,ignore_index = True)
    print("Thank you for the input. Your work is finished!")

name_feedback()

export_girl_csv = girl_name_feedback.to_csv(params.local+'/girl_name_feedback.csv', index = None, header=True)
export_boy_csv = boy_name_feedback.to_csv(params.local+'/boy_name_feedback.csv', index = None, header=True)