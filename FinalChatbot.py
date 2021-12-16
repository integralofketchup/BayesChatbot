# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util

import numpy as np
import re
from porter_stemmer import PorterStemmer
import math
import random


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.creative = creative
        self.ps = PorterStemmer()

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        sent_dict = {}
        for word in self.sentiment:
            stem = self.ps.stem(word, 0, len(word) - 1)
            sent_dict[stem] = self.sentiment[word]
        self.sentiment = sent_dict
        self.negations = ["not", "never", "didn't", "won't", "no"]
        self.responses = 0
        self.recommend_count = 0
        self.recommendations = []
        self.emph = ['very', 'really', 'extremely', 'reeally']
        self.pos = ['love', 'loved', 'amazing', 'great']
        self.neg = ['terrible', 'horrible', 'hate', 'hated']

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        self.user_ratings = np.zeros(shape = len(self.titles))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hi! I'm MovieBot! I'm going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Thank you for hanging out with me! Stay in touch! Goodbye!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
            #find movies by title creatives - 12 points
            movies = self.find_movies_by_title(line)
            if len(movies) > 0:
                print(movies)
            #disambiguation 2 - 4 points
            #disambiguation 3 - 4 points
            if '[' in line and ']' in line and line[line.find('[')+1].isnumeric():
                title = self.extract_titles(line)
                #print(title[0])
                #print(line[line.find('['):line.find(']')+1])
                print(self.disambiguate(title[0], line[line.find('['):line.find(']')+1]))
            movies = self.extract_sentiment_for_movies(line)
            if len(movies) == 2:
                if movies[0][1] > 0 and movies[1][1] > 0:
                    return "You liked " + movies[0][0] + " and " + movies[1][0]
                elif movies[0][1] < 0 and movies[1][1] < 0:
                    return "You disliked " + movies[0][0] + " and " + movies[1][0]
                elif movies[0][1] > 0 and movies[1][1] < 0:
                    return "You liked " + movies[0][0] + " but disliked " + movies[1][0]
                elif movies[0][1] < 0 and movies[1][1] > 0:
                    return "You liked " + movies[1][0] + " but disliked " + movies[0][0]
                elif movies[0][1] == 0 and movies[1][1] == 0:
                    return "Seems like you don't have a strong opinion on any of the movies."
                elif movies[0][1] == 0 and movies[1][1] < 0:
                    return "You disliked " + movies[1][0]
                elif movies[0][1] == 0 and movies[1][1] > 0:
                    return "You liked " + movies[1][0]
                elif movies[0][1] < 0 and movies[1][1] == 0:
                    return "You disliked " + movies[0][0]
                elif movies[0][1] > 0 and movies[1][1] == 0:
                    return "You liked " + movies[0][0]

        else:
            if self.responses > 4:
                if len(self.recommendations) < self.recommend_count + 2:
                    return "Sorry, there are no more recommendations. Please type :quit to exit."
                if line.lower() == "yes":
                    self.recommend_count += 1 
                    recommended = self.titles[self.recommendations[self.recommend_count]][0]

                    responses = [
                        "Based on your earlier reviews, I recommend " + recommended + "for you! Would you like more recommendations? (Yes/ No)",
                        "According to your response, I recommend " + recommended + "! Any further recommendations? (Yes/ No)",
                        "I think you would love " + recommended + " by your reviews! Interested in more recommendations? (Yes/ No)"
                    ]
                    return random.choice(responses)
                elif line.lower() == "no":
                    return "You indicated that you would not like any further recommendations. Please type :quit to exit."
                return "Please strictly indicate if you would like further recommendations (Yes/ No)."

            potential = self.extract_titles(line)
            no_quotations = [
                "I'm sorry, I didn't catch that. Please put the movie in quotations.",
                "Come again? Make to to put your movie in quotations!",
                "I didn't register that. Please make sure that your request is in quotations!",
            ]
            if len(potential) == 0:
                return random.choice(no_quotations)
            mov = potential[0]
            movies = self.find_movies_by_title(mov)
            
            none_found = [
                "We didn't find any results for " + mov + ". Name some other film please!", 
                "The movie " + mov + " does not exist within our database. Tell me about another movie instead!", 
                "It appears that" + mov + " does not exist within our database. What about other movies?",
                "I'm unfamiliar with " + mov + ", unfortunately. What about a different movie?",
                "It seems " + mov + " is not in my database, Perhaps provide a different film?"
            ]
            multiple_options = [
                "Sorry there were multiple titles for " + mov + ", please name a specific instance."
            ]
            first_result = None  
            if len(movies) == 0: 
                closest_movies = self.find_movies_closest_to_title(mov)
                if len(closest_movies) == 1:
                    selection = self.titles[closest_movies[0]][0]
                    response = input("Did you mean: " + selection)
                    first_result = closest_movies[0] if response.lower() == "yes" else first_result
                elif len(closest_movies) > 1: 
                    selections = self.titles[closest_movies[0]][0]
                    for i in range(1, len(closest_movies)):
                        selections = selections + ", " + self.titles[closest_movies[i]][0] 
                    response = input("Did you mean any of these movies: " + selections + ". If so, state which one!")
                    response = self.disambiguate(response, closest_movies)
                    first_result = closest_movies[0] if response else first_result
                if not first_result: 
                    return random.choice(none_found)
            elif len(movies) == 1:
                first_result = movies[0]
            else:
                return random.choice(multiple_options)
            sentiment = self.extract_sentiment(line)
            
            unclear = [
                "It's unclear whether or not you enjoyed " + mov + ", please explain further!",
                "I'm not sure whether you liked or disliked ' " + mov + ", please clarify.",
                "Tell me more about ' " + mov + ". I'm not sure of your position on it.", 
                "Let me know more about ' " + mov + ". Not sure what you really think of it.",
                "I can't tell if you enjoyed ' " + mov + ". Care to explain further?"
            ]
            if sentiment == 0:
                return random.choice(unclear)

            last = self.user_ratings[first_result]
            self.user_ratings[first_result] = sentiment
            recent = self.user_ratings[first_result]
            if last != recent:
                self.responses = self.responses + 1
            if self.responses > 4: 
                self.recommendations = self.recommend(self.user_ratings, self.ratings)
                #Grab the first recommendation
                recommended = self.titles[self.recommendations[0]][0]
                return "Based on your earlier reviews, I recommend " + recommended + "for you!"

            positive_choices = [
                "I also loved " + mov + "! Please name another movie which you liked!",
                "Nice! I'm glad that you enjoyed watching " + mov + "! What about another one?",
                "Oh wow! I guess I'll look into " + mov + "! Do you have any other opinions?",
                "A great choice. " + mov + " seemed enjoyable. How about another one?",
                "Oh cool! I'll look into " + mov + " then! What's your view on another movie?"
            ]
            
            negative_choices = [
                "I'm sorry that you don't like " + mov + ". What did you think of other movies?",
                "Darn, I guess " + mov + " wasn't that great. Do you have any thoughts on other movies?",
                "Oh well. It seems" + mov + " wasn't the best. Do you have any other opinions?",
                "Aw shucks. It seems" + mov + " wasn't a hit. Have any other reviews?",
                "Got it. " + mov + " is a bad egg. Have another one in the basket?"
            ]

            if sentiment > 0:
                return random.choice(positive_choices)
            else:
                return random.choice(negative_choices) 

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        movies = []
        pattern = '"(.*?)"'
        matches = re.findall(pattern, preprocessed_input)
        for match in matches: 
            movies.append(match)
        return movies

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        if self.creative:
            articles = ["a", "an", "the", "la", "las", "el", "los"]
            title = title.lower()
            title = title.strip('"')
            titletok = title.split(" ")
            end = ""
            begin = ""
            year = ""
            #formats stuff i.e. the president (2090) => president, the (2090)
            for word in titletok:
                if word in articles:
                    end += word + " "
                elif '(' in word and ')' in word and word.replace('(','').replace(')','').isnumeric():
                    year += word
                else:
                    begin += word + " "

            if len(begin) > 0:
                begin = begin[:-1]
            if len(end) > 0:
                end = end[:-1]

            title = begin if len(end) == 0 else begin + ", " + end
            title += " " + year if len(year) > 0 else ""
            #find match
            matches = []
            for i in range(len(self.titles)):
                if len(year) > 0:
                    if title in self.titles[i][0].lower():
                        matches.append(i)
                #if no year specified, strip database titles of year i.e. scream (2040) => scream
                #find exact match else stuff like "screamer" works for input scream 
                else:
                    loc = self.titles[i][0].find('(')
                    t = ""
                    if (self.titles[i][0][loc+1].isnumeric()):
                        t = self.titles[i][0][:loc+1]
                        t = t[:-1]
                        t = t[:-1]
                    else:
                        t = self.titles[i][0]
                    #print(t.lower())
                    #scream - eliminate screamer, accept scream)
                    if title in t.lower():
                        extraletter = t.lower().find(title)+len(title)
                        if extraletter < len(t):
                            if not t.lower()[extraletter].isalpha():
                                matches.append(i)
                        else:
                            matches.append(i)
            return matches
        else:
            articles = ["a", "an", "the"]
            title = title.lower()
            title = title.strip('"')
            titletok = title.split(" ")
            end = ""
            begin = ""
            year = ""
            #formats stuff i.e. the president (2090) => president, the (2090)
            for word in titletok:
                if word in articles:
                    end += word + " "
                elif word.replace('(','').replace(')','').isnumeric():
                    year += word
                else:
                    begin += word + " "

            if len(begin) > 0:
                begin = begin[:-1]
            if len(end) > 0:
                end = end[:-1]

            title = begin if len(end) == 0 else begin + ", " + end
            title += " " + year if len(year) > 0 else ""
            #find match
            matches = []
            for i in range(len(self.titles)):
                if len(year) > 0:
                    if title == self.titles[i][0].lower():
                        matches.append(i)
                #if no year specified, strip database titles of year i.e. scream (2040) => scream
                #find exact match else stuff like "screamer" works for input scream 
                else:
                    loc = self.titles[i][0].find('(')
                    t = self.titles[i][0][:loc+1]
                    t = t[:-1]
                    t = t[:-1]
                    #print(t.lower())
                    if title == t.lower():
                        matches.append(i)
        return matches

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        cleaned_input = preprocessed_input
        movies = self.extract_titles(preprocessed_input)
        for movie in movies:
            cleaned_input = cleaned_input.replace(movie, "")

        if cleaned_input == "": 
            return 0

        cleaned_input = cleaned_input.lower()
        tokens = [self.ps.stem(w, 0, len(w) - 1).lower() for w in cleaned_input.split()]

        negative = False
        emph = False
        overall_sent = 0
        for tok in tokens:
            tok = tok.strip(" .?!")
            if tok in self.negations:
                negative = True
            if tok in self.emph:
                emph = True
                            
            sentiment = 0
            if tok in self.pos:
                sentiment = 2
            elif tok in self.neg:
                sentiment = -2
            elif tok not in self.sentiment:
                continue
            else:
                sentiment = 1 if self.sentiment[tok] == "pos" else -1
                if emph: sentiment *= 2
            emph = False
            overall_sent = overall_sent + sentiment if not negative else overall_sent - sentiment
            negative = False
        overall_sent = max(overall_sent, -2)
        overall_sent = min(overall_sent, 2)
        return overall_sent

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        sentList = []
        # butList = [m.start() for m in re.finditer('but', preprocessed_input)]
        
        movies = self.extract_titles(preprocessed_input)
        sent = self.extract_sentiment(preprocessed_input)
        
        cleaned_input = preprocessed_input.lower()   
            
        for movie in movies:
            movieIndex = cleaned_input.find(movie.lower())
            butIndex = cleaned_input.find('but')
            if butIndex < movieIndex and butIndex != -1:
                sentList.append((movie, sent * -1))
            else:
                sentList.append((movie, sent))                
        return sentList
        return 0




    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        title_length = len(title)
        title = title.lower()
        options = {}
        for option in range(len(self.titles)):
            comparison = self.titles[option][0].lower()
            if "(" in comparison:
                index = comparison.index("(")
                comparison = comparison[:index - 1]
            comparison_length = len(comparison)
    
            lev_array = [[0 for i in range(comparison_length + 1)] for j in range(title_length + 1)] 
            for i in range(title_length + 1):
                lev_array[i][0] = i
            for i in range(comparison_length + 1):
                lev_array[0][i] = i
            for i in range(1, title_length + 1):
                for j in range(1, comparison_length + 1):
                    deletion = lev_array[i - 1][j] + 1
                    insertion = lev_array[i][j - 1] + 1
                    substitution = lev_array[i - 1][j - 1] if title[i - 1] == comparison[j - 1] else lev_array[i - 1][j - 1] + 2
                    lev_array[i][j] = min(deletion, insertion, substitution)

            if max_distance >= lev_array[title_length][comparison_length]:
                options[option] = lev_array[title_length][comparison_length]

        minimum = []
        if len(options) == 0:
            return minimum
        dist_list = sorted([pair for pair in options.items()], key = lambda x: x[1])

        min_edit = dist_list[0][1]
        for i in range(len(dist_list)):
            if dist_list[i][1] > min_edit: 
                break
            minimum.append(dist_list[i][0])
        return minimum
        

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        title = clarification
        movielist = candidates
        articles = ["a", "an", "the", "la", "las", "el", "los"]
        title = title.lower()
        title = title.strip('"')
        titletok = title.split(" ")
        end = ""
        begin = ""
        year = ""
        #formats stuff i.e. the president (2090) => president, the (2090)
        for word in titletok:
            if word in articles:
                end += word + " "
            elif '(' in word and ')' in word and word.replace('(','').replace(')','').isnumeric():
                year += word
            else:
                begin += word + " "

        if len(begin) > 0:
            begin = begin[:-1]
        if len(end) > 0:
            end = end[:-1]

        title = begin if len(end) == 0 else begin + ", " + end
        title += " " + year if len(year) > 0 else ""
        #find match
        matches = []
        for i in movielist:
            #year
            i = int(i)
            if len(title) == 4 and title.isnumeric():
                if title in self.titles[i][0].lower():
                    matches.append(i)
            else:
                loc = self.titles[i][0].find('(')
                t = ""
                if (self.titles[i][0][loc+1].isnumeric()):
                    t = self.titles[i][0][:loc+1]
                    t = t[:-1]
                    t = t[:-1]
                else:
                    t = self.titles[i][0]
                #print(t.lower())
                #scream - eliminate screamer, accept scream)
                if title in t.lower():
                    extraletter = t.lower().find(title)+len(title)
                    if title.isnumeric():
                        if extraletter < len(t):
                            if not t.lower()[extraletter].isalpha():
                                if not t.lower()[extraletter] == ')' or not t.lower()[extraletter].isnumeric():
                                    matches.append(i)
                        else:
                            matches.append(i)
                    else:
                        if extraletter < len(t):
                            if not t.lower()[extraletter].isalpha():
                                matches.append(i)
                        else:
                            matches.append(i)

        if len(matches) == 0:
            if(title.isnumeric() and int(title) < len(movielist)):
                matches.append(movielist[int(title)-1])
            if("recent" in title.lower()):
                recent = 0
                ind = 0
                for i in movielist:
                    i = int(i)
                    loc = self.titles[i][0].find('(')
                    year = int(self.titles[i][0][loc+1: loc+5])
                    #print(year)
                    if(year >= recent):
                        ind = i
                        recent = year
                matches.append(ind)

        return matches

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros(ratings.shape)
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[ratings <= threshold] = -1
        binarized_ratings[ratings == 0] = 0

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        movies = []
        for i in range(len(user_ratings)):
            if user_ratings[i] != 0:
                movies.append(i)

        scores = {}
        for r in movies:
            for m in range(len(ratings_matrix)):
                if r == m or m in movies: 
                    continue
                similarity = user_ratings[r] * self.similarity(ratings_matrix[r], ratings_matrix[m])
                scores[m] = similarity if m not in scores else scores[m] + similarity

        all_recommends = [pair for pair in list(scores.items()) if not math.isnan(pair[1])]      
        all_recommends = sorted(all_recommends, key = lambda x: x[1], reverse = True)
        recommendations = []
        for i in range(k):
            recommendations.append(all_recommends[i][0])

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')