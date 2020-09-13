# Read in the data
import csv
with open('us-states.csv') as file:
        state_covid_data = []
        for row in csv.reader(file):
            state_covid_data.append(row)

#how many total observations?
len(state_covid_data)


## (1) Count up the number of unique dates in the data.
unique_dates = set([inner_list[0] for inner_list in state_covid_data])
print(unique_dates)


## (2) Find the first date in which the District of Columbia recorded a case.
for row in state_covid_data:
    if 'District of Columbia' in row:
        print(row[0])
        break


## (3) Write a function that takes in a state name as input (e.g. "Wisconsin") and outputs the date of its first case.
def first_covid_date(state):
    for row in state_covid_data:
        if state in row:
            print(row[0])
            break


#let's test the function
first_covid_date('Wisconsin')
first_covid_date('Illinois')



## (Optional) Bonus: Write a function that takes in a state name as input (e.g. "Wisconsin") and outputs the date when the number of reported cases within the state exceeded 1000.
def thousand_covid_date(state):
    sum_of_cases = []
    for row in state_covid_data:
        if row[0] == 'date':
            continue
        if state in row and sum(sum_of_cases) < 1000:
            sum_of_cases.append(int(row[3]))
            #print(sum(sum_of_cases))
        if state in row and sum(sum_of_cases) >= 1000:
            print(row[0])
            break

#testing the function
thousand_covid_date('Wisconsin')
thousand_covid_date('Hawaii')
