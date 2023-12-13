

import operator

#1:
def test_print():
    print "This is a test statement."
if __name__ == '__main__':
    test_print()

#2:
def list_set_length():
    items_list = [1, 2, 3, 4, 3, 2, 1]
    items_set = {1, 2, 3, 4, 3, 2, 1}
    print 'The size of the list is: ', len(items_list)
    print 'The size of the set is: ', len(items_set)

#3:
def intersection():
    S = {1,2,3,4}
    T = {3,4,5,6}
    print [x for x in S if x in T]

#4:
S = {-4,-2,1,2,5,0}
def three_tuples():
    tuple_list = [ (x,y,z) for x in S for y in S for z in S if x+y+z == 0]
    print tuple_list

#5a:
def dict_init():
    mydict= {'Neo':'Keanu', 'Morpheus':'Laurence', 'Trinity':'Carrie-Anne'}
    print "Initialized dictionary :"
    print mydict

#5b:
def dict_find(dlist,key):
    find_list = [ x[key] if key in x else 'NOT PRESENT' for x in dlist   ]
    print "Key: ",key
    print "Corresponding list:"
    print find_list

#6:
def file_line_count():
    line_count = sum(1 for line in open('stories.txt'))
    print "The number of lines:"
    print line_count

#7a:
def make_inverse_index(strlist):
    mydict = {}
    for line, story in strlist.items():
        for word in story.lower().split():
            if mydict.get(word,False):
                if line not in mydict[word]:
                    mydict[word].append(line)
            else:
                mydict[word] = [line]
    return mydict

#7b:
def or_search(inverseIndex,query):
    result_set = set()
    for word in query:
        word_set = set()
        if word in inverseIndex:
            word_set = set(inverseIndex[word])
            for index in word_set:
                result_set.add(index)
    return result_set

#7c:
def and_search(inverseIndex,query):
    result_set = set()
    for word in query: 
        if result_set == set():
            result_set = set(inverseIndex[word])
        else:
            result_set = set(inverseIndex[word]) & result_set
    return result_set

#7d:
def most_similar(inverseIndex,query):
    list_or = list(or_search(inverseIndex,words))
    line_score = []
    mostSimilarSet = set()
    for line in list_or:
        score = 0
        for word in query:
            if line in inverseIndex[word]:
                score += 1
        line_score.append(score)                
    score_dict = dict(zip(list_or, line_score))
    return sorted(score_dict.items(), key=operator.itemgetter(1), reverse = True)





if __name__ == '__main__':
    print "Question 1:"
    test_print()
    print "Question 2: "
    list_set_length()
    print "Question 3:"
    intersection()
    print "Question 4:"
    three_tuples()
    print "Question 5a:"
    dict_init()
    print "Question 5b:"
    dlist= [{'Neo':'Keanu', 'Morpheus':'Laurence', 'Trinity':'Carrie-Anne'},
             {'Neo':'Keanub', 'Trinity':'Carrie-Anneb'}]
    key='Neo'
    dict_find(dlist,key)
    print "Question 6:"
    file_line_count()
    print "Question 7a:"
    inputlist=dict(enumerate(open('stories.txt')))
    inverseIndex = make_inverse_index(inputlist);
    print inverseIndex
    print "Question 7b:"
    words = ['runaways','volumes']
    result_or = or_search(inverseIndex,words)
    print result_or
    print "Question 7c:"
    words = ['other','june','all-you-can']
    result_and = and_search(inverseIndex,words)
    print result_and
    print "The index and the corresponding number of keywords(contained in the document):"
    print "Lipeng: Todo"

#------------------------------------------------------------------------
#Question 1:
#This is a test statement.
#Question 2: 
#The size of the list is:  7
#The size of the set is:  4
#Question 3:
#[3, 4]
#Question 4:
#[(0, 0, 0), (0, 2, -2), (0, -2, 2), (1, 1, -2), (1, -2, 1), (2, 0, -2),
# (2, 2, -4), (2, -4, 2), (2, -2, 0), (-4, 2, 2), (-2, 0, 2), (-2, 1, 1),
# (-2, 2, 0)]
#Question 5a:
#Initialized dictionary :
#{'Neo': 'Keanu', 'Trinity': 'Carrie-Anne', 'Morpheus': 'Laurence'}
#Question 5b:
#Key:  Neo
#Corresponding list:
#['Keanu', 'Keanub']
#Question 6:
#The number of lines:
#50
#Question 7b:
#set([0, 17, 15])
#Question 7c:
#set([13])
#Question 7d:
#The index and the corresponding number of keywords(contained in the document):


