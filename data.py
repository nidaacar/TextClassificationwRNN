train_data = {
    'the movie was great': True,
    'this is not at all bad': True,
    'this is good right now': True,
    'this is sad right now': False,
    'this is very bad right now': False,
    'this was good earlier': True,
    'i was not happy and not good earlier': False,
    'the soundtrack of the movie was very bad': False,
    'screenplay and acting are very unsuccessful.': False,
    'visuality and subject is very bad.': False,
    'a very boring movie.': False,
    'a very high quality movie.': True,
    'soundtrack was very nice.': True,
    'I wish I had not watched.': False,
    'Another movie can be watched rather than watching this.': False,
    'every scene was a great movie.': True,
    'every scene was a successful movie.': True,
    'a movie that must be watched.': True,
    'I strongly recommend.': True,
    'it would be a more successful movie if another actress played the leading role.': False,
    'successful performance.': True,
    'has a very high quality staff.': True,
    'no acting': False,
    'Not successfully adapted from the book': False,
    'it was not funny': False,
    'I never laughed': False,
    'I got bored from start to finish in the movie.': False,
    'If anyone has not watched yet, watch it right away.': True,
    'a super movie.': True,
    'This movie is a movie masterpiece': True,
    'best movie ever': True,
    'It would be a more successful film if another actor played the male lead role.': False,
    'there were too many unnecessary scenes.': False,
    'I will definitely not recommend it to anyone.': False,
    'I never laughed in the movie': False,
    'a disgraceful movie': False,
    'my hours were wasted': False,
    'a masterpiece': True,
    'The fiction of the movie was very successful.': True,
    'an exceptional movie.': True,
    'The desire to be told could not be exhibited successfully': False,
    'good': True,
    'bad': False,
    'happy': True,
    'sad': False,
    'not good': False,
    'not bad': True,
    'not happy': False,
    'not sad': True,
    'very good': True,
    'it is not enough': False,
    'he is a good boy': True,
    'i did not  like it': False,
    'i am going to take it': True,
    'i am not proud of you': False,
    'she was a kind person': True,
    'this is not make any sense': False,
    'let say x is equal to 1': True,
    'my dog is really smart': True,
    'i missed my friend': True,
    'this shirt is not large': False,
    'she catched the sunlight ': True,
    'this skirt is too expensive': True,
    'meal was not delicious': False,
    'we will see good days': True,
    'this movie will come to theater very soon': True,
    'do not step on the grass': False,
    'he keeps a promise to his son': True,
    'we will never forget these days': True,
    'You won’t need to register or leave any details to access the dataset': False,
    'Which is diagnosed more often in America (2011)?': True,
    'City has a higher average altitude': True,
    'She sold more albums while living': True,
    'very bad': False,
    'very happy': True,
    'very sad': False,
    'i am happy': True,
    'this is good': True,
    'i am bad': False,
    'this is bad': False,
    'i am sad': False,
    'this is sad': False,
    'i am not happy': False,
    'this is not good': False,
    'i am not bad': True,
    'this is not sad': True,
    'i am very happy': True,
    'this is very good': True,
    'i am very bad': False,
    'this is very sad': False,
    'this is very happy': True,
    'i am good not bad': True,
    'this is good not bad': True,
    'i am bad not good': False,
    'i am good and happy': True,
    'this is not good and not happy': False,
    'i am not at all good': False,
    'i am not at all bad': True,
    'i am not at all happy': False,
    'this is not at all sad': True,
    'this is not at all happy': False,
    'i am good right now': True,
    'i am bad right now': False,
    'this is bad right now': False,
    'i am sad right now': False,
    'i was good earlier': True,
    'i was happy earlier': True,
    'i was bad earlier': False,
    'i was sad earlier': False,
    'i am very bad right now': False,
    'this is very good right now': True,
    'this is very sad right now': False,
    'this was bad earlier': False,
    'this was very good earlier': True,
    'this was very bad earlier': False,
    'this was very happy earlier': True,
    'this was very sad earlier': False,
    'i was good and not bad earlier': True,
    'i was not good and not happy earlier': False,
    'i am not at all bad or sad right now': True,
    'i am not at all good or happy right now': False,
    'this was not happy and not good earlier': False,
    'the movie is amazing': True,
    'rated very well': True,
    'The script of the movie is very bad.': False,
    'I wasted my time in vain.': False,
    'music selection failed.': False,
    'It has a very boring subject.': False,
    'the acting is very bad.': False,
    'I dont think I will watch it again.': False,
    'I did not love.': False,
    'the lead actor is very good': True,
    'the lead actress is very good': True,
    'the lead actor is very good': True,
    'child actors were very successful': True,
    'the actress was very successful': True,
    'the actor was very successful': True,
    'the acting is great': True,
    'I made a very wrong choice.': False,
    'the movie has a bad subject.': False,
    'the subject of the movie was very nice': True,
    'the end of the movie was very nice': True,
    'I was very surprised at the end of the movie, it was very nice': True,
    'I cried a lot at the end of the movie, it was very nice': True,
    'it was a very successful movie.': True,
    'the movie impressed me but they think is a normal movie.': True,
    'the director of the movie was very good.': True,
    'The screenwriter of the movie is great.': True,
    'The director of the movie is very successful.': True,
    'a low-budget nonsense movie': False,
    'I did not like the movie': False,
    'very simple movie': False,
    'The subject of the movie is very successful.': True,
    'The movie is so boring': False,
    'the movie was very gripping.': True,
    'The selection of the cast was very good.': True,
    'It was one of the best movies I have seen.': True,
    'It was the most beautiful movie I have ever seen in my life.': True,
    'The fiction of the movie is very bad.': False,
    'a failed movie.': False,
    'player selection could have been better.': False,
    'I never liked this movie.': False,
    'The male lead actor of the movie is very unsuccessful.': False,
    'I was so bored watching the movie.': False,
    'the most beautiful movie in the world': True,
    'the subject of the movie is very impressive': True,
    'best of its kind': True,
    'a movie that everyone should watch.': True,
    'I was never bored watching.': True,
    'a perfect movie.': True,
    'a masterfully made movie': True,
    'the movie confused me.': False,
    'I did not understand anything from the movie.': False,
    'I closed the movie in the middle.': False,
    'it was a technically flawless movie.': True,
    'the actors performed tremendously.': True,
    'wrong choice.': False,
    'I stopped watching without waiting for the end of the movie.': False,
    'its a very mediocre movie': False,
    'the actors have played very contrivedly.': False,
    'player selection could have been better.': False,
    'terrible acting.': False,
    'the light is very bad.': False,
    'Visualizations and effects were terrible.': False,
    'it was not good.': False,
    'he had a simple script.': False,
    'It was a simple film made on a low budget.': False,
    'I was not interested in the subject.': False,
    'I do not recommend.': False,
    'a waste of time.': False,
    'I liked it so much that I can watch it again.': True,
    'screenplay and acting are very successful.': True,
    'visuality and subject is very good.': True,
    'a very successful movie.': True,
    'the director is very unsuccessful.': False,
    'the book was not successfully adapted to the film': False,
    'It was a terrible movie from start to finish.': False,
    'I wish i did not watch': False,
    'I watched the movie bored': False,
    'player selection was very wrong': False,
    'a very well film.': True,
    'scenes were shot very skillfully.': True,
    'Tied for the best movie I have ever seen': True,
    'The only other movie I have ever seen that effects me as strongly is To Kill a Mockingbird': False,
    'I didnt intend to see this movie at all: I do not like prison movies and I dont normally watch them': False,
    'No action, no special effects - just men in prison uniforms talking to each other': False,
    'The Shawshank Redemption and To Kill a Mockingbird are the best movies I have ever seen': True,
    'Dark, yes, complex, ambitious. Christopher Nolan and his co-writer Jonathan Nolan deserve a standing ovation': True,
    'This is a masterpiece. A timeless masterpiece': True,
    'I didnt like this film all that much - I found it rather over-hyped and boring': False,
    'I feel that this film has not dated all that much and has tremendous re-watch-ability': True,
    'Simply put, this movie changed my life': True,
    'This movie is literally the first time I ever came upon something that, at first sight seemed incredibly stylish, sophisticated and entertaining': True,
    'David Fincher, director, was probably the only reason I went to see this movie in the first place': True,
    'This movie rates as one of my all-time favorite movies and, simply': True,
    'Ive just re-watched The Lord of the Rings trilogy for the 1000th time tonight': True,
    'I miss the good old LOTRs days. The best movies ever created': True,
    'The casting is perfection as well as the incredible acting by everyone in the movie': True,
    'Tarantino is without a doubt one of the best directors of all time and maybe the best of the 90s': True,
    'movies like Tarantinos or the Shawshank Redemption deserved much more': True,
    'On a partial first viewing, I didnt like The Good, the Bad and the Ugly': False,
    'In my opinion, 12 Angry Men is the greatest film that has ever been created': True,
    'A perfect example of less is more': True,
    'Im not in agreement that Inception is a classic perfect film but it is a very good one': True,
    'The actors in the movie are very good': True,
    'The place where the movie was shot is very beautiful': True,
    'The directors fiction is very good': True,
    'This movie is a masterpiece': True,
    'This movie was so emotional': True,
    'This movie was so scary and scary': False,
    'I dont like horror movies': False,
    'i dont like this movie': False,
    'Characters were incompetent': False,
    'Acting performance was low': False,
    'People were not compatible with each other': False,
    ' The film Ive been waiting so much from the early days of the shooting': True,
    'Artificial intelligence fiction in the Metrix movie was very good': True,
    'The movie was simply amazing': True,
    'The shooting of this movie at that time amazed me': True,
    'Metrix character was intertwined with the actor': True,
    'The character was very good': True,
    'The fighting scenes were amazing.': True,
    'The editing was great.': True,
    'All films of this director Lana Wachowski are very successful': True,
    'I like adventure movies. Thats why I like this movie.': False,
    'The script of this movie was ridiculous': False,
    'Characters did not master the script': False,
    'Emotion was not reflected in the movie.': False,
    'This movie did not meet my expectation.': False,
    'I regret that I watched this movie.': False,
    'This wasted time wasted for me.': False,
    'The acting of the girl impressed me in the movie.': True,
    'It made me feel the most beautiful form of love.': True,
    'It was an emotional movie.': True,
    'He was telling true love. Thats why I loved it.': True,
    'It was one of the funniest movies I have seen.': True,
    'My companion made me laugh a lot.': True,
    'Acting was legendary.': True,
    'It was one of the movies I watched with pleasure.': True,
    'but The Green Mile is an incredibly effective prison drama with terrific performances ': True,
    'Needs to be seen to be believed; in one word: perfection.': True,
    'Ill never get tired of watching Goodfellas': True,
    'the entertainment value of this film is just amazing.': True,
    'A simple story becomes a sad and poignant movie about ordinary people.': True,
    'Every person I know who had the chance to watch this wonderful movie have cried -especially during the second half.': True,
    'Really brilliant work, presenting some top-notch performances': True,
    'This movie led me to think differently.': False,
    'I didnt like it because it is a thriller': False,
    'They could not fully reflect the feeling to the film.': False,
    'acting was poor.': False,
    'The script was not what I expected.': False,
    'Im disappointed.': False,
    'Being french and a film maker myself, I have high standards for ratings, and this one definitely deserves in 10/10.': True,
    'Ive not seen a film showing our world with such humour in a long time.': True,
    'The jokes are absurd and possibly, with a touch of British humour to them.': True,
    'The directing is beautiful': True,
    'the acting is incredible': True,
    'the shots are somehow truthful': True,
    'How a teacher touches a childs life': True,
    'The value the teacher gives to his students is very well explained.': True,
    'He explains that the best touch in your life is to come across a good teacher.': True,
    'The lead actor was a professional.': True,
    'The lead actor played very badly.': False,
    'The choice of music in the movie was successful.': True,
    'The dress selection of the actors corresponded to that period.': True,
    'The sound effect in the movie was very successful.': True,
    'The subject and actors of the movie were complementary.': True,
    'The duration of the movie was not long enough to squeeze the audience.': True,
    'The selected subject of the movie made a big impact on the audience.': True,
    'Some of his scenes were quite realistic.': True,
    'The subject of the movie entertained the audience.': True,
    'At the end of the movie, the audience was taught.': True,
    'The subject of the film explained both the period and aroused a great impact on the audience.': True,
    'The choice of music in the movie was very bad.': False,
    'The dress choice of the players did not match that period.': False,
    'The sound effect in the movie was quite unsuccessful.': False,
    'The subject and actors of the movie did not complement each other.': False,
    'The duration of the movie was too long.': False,
    'The selected topic of the film did not leave the desired effect on the audience.': False,
    'Some of his scenes were quite fiction.': False,
    'Although the subject of the movie is comedy, he never entertained.': False,
    'At the end of the movie, he could not find what he hoped for the audience.': False,
    'The subject of the film was independent of the period described.': False,
    'the movie was so emotional': True,
    'It is among the top 100 movies to watch.': True,
    'the movie appeals to all ages.': True,
    'The film is main idea was very good.': True,
    'the subject of the movie was legend': True,
    'the actors played very well.': True,
    'behind the scenes was very good.': True,
    'i will watch again': True,
    'it was great': True,
    'the movie was legend': True,
    'I wish I had watched it before.': True,
    'It was spectacular': True,
    'this movie cannot be awarded': False,
    'I do not recommend, do not watch': False,
    'was very good.': True,
    'I m tired of laughing': True,
    'It was so funny': True,
    'that was so fun.': True,
    'everything was very nice in the movie': True,
    'it managed to attract my attention.': True,
    'I regret watching what I watched': False,
    'I was not interested in the events': False,
    'I watched with curiosity until the end': True,
    'the end is amazing': True,
    'it was a curious movie': True,
    'I wish the sequel would come': True,
    ' I was never scared': False,
    'how was this horror movie': False,
    'this horror looks more like a funny movie than the movie': False,
    'i never loved this movie': False,
    'the series of this movie should be made': True,
    'I watched it very fondly': True,
    'i love the movie': True,
    'I do not mind': False,
    'They will be joining us for dinner tonight': True,
    'When I find myself in times of trouble': True,
    'Mother Mary comes to me': True,
    'Speaking words of wisdom, let it be': True,
    'There will be an answer, let it be': True,
    'If I am not back again this time tomorrow': False,
    'Goodbye everybody, I have got to go': True,
    'I sometimes wish I would never been born at all': True,
    'Everyday (everyday) I try and I try and I try': True,
    'But everybody wants to put me down': True,
    'It is not easy love': False,
    'But you have got friends you can trust': True,
    'When you are in need of love they give you care and attention': True,
    'I have paid my dues': True,
    'I have had my share of sand kicked in my face,but I have come through': True,
    'this made me think that perhaps two-choice questions are better than true/false questions': True,
    'there is not really a well-known base rate': False,
    'I consider it a challenge before the whole human race': True,
    'I am not flying to England': False,
    'They are not from Ecuador': False,
    'Maybe January light will consume': True,
    'In this part of the story he is not the one who dies': False,
    'Now you know the benefits of listening to music in English': True,
    'The player will not get trained on a base rate of 50%': False,
    'Intelligent psychological drama was awesome': True,
    'it is not significant without being overstated ': False,
    'you can not put down ': False,
    'you would not call the good girl a date movie': False,
    'it is hard to know what to praise first': True
}

test_data = {
    'this is happy': True,
    'i am good': True,
    'this is not happy': False,
    'i am not good': False,
    'this is not bad': True,
    'i am not sad': True,
    'i am very good': True,
    'this is very bad': False,
    'i am very sad': False,
    'this is bad not good': False,
    'this is good and happy': True,
    'i am not good and not happy': False,
    'i am not at all sad': True,
    'this is not at all good': False,
    'I did not understand what the movie wanted to tell': False,
    'the end of the movie is uncertain.': False,
    'I could not wait for the end of the movie': False,
    'I do not like': False,
    'it was not nice': False,
    'Anafikir to be released from the movie is great': True,
    'It has a great subject.': True,
    'a beautiful movie that amazes you.': True,
    'the cast is very unskilled.': False,
    'The female lead actor of the movie is very unsuccessful.': False,
    'The music selection of the movie was very successful.': True, 'I laughed a lot while watching the movie': True,
    'I had a lot of fun watching the movie': True,
    'it was an extraordinary movie': True,
    'I recommend this movie to everyone': True,
    'the acting is great': True,
    'I wish there was not a movie like this': False,
    'The film was filmed in the dark area and bored me.': False,
    'I m glad I watched': True,
    'oscara nominated movie': True,
    'this movie gets a definite award': True,
    'players are very good': True,
    'the movie was shot in the same place, it was bad.': False,
    'the script of the movie was terrible.': False, 'this one is really a good one': True,
    'I shall never now be satisfied': False,
    'i supposed to go out': True,
    'when i am not supposed to selena gomez': False,
    'she is stuck in his head': True,
    'Cause it is not me': False,
    'I walked across an empty land': True,
    'I knew the pathway like the back of my hand': True,
    'I felt the earth beneath my feet': True,
    'Oh, simple thing, where have you gone?': True,
    'I am getting old, and I need something to rely on': True,
    'We will never support discrimination': False,
    'We are not responsible for this part': True,
    'I do not see how you can': False,
    'I try to stay awake and remember my name': True,
    'I do not feel the same': False,
    'Everybody is changing': True,
    'This is the last time': True,
    'negative sentences are not necessarily a bad thing': False,
    'Let is take a closer look at negative statement constructs': True,
    'However, for informal writing, including blogging and social media posts, contractions are perfectly acceptable.': True,
    'he did not like Bikhram yoga': False,
    'They will not be joining us for dinner tonight': False,
    'She will not be attending the Met Gala this year': False,
    'We were happy when he moved away': True,
    'Something I was not sure of': False,
    'And years make everything alright': True,
    'I agree that positive sentences can make our writing clearer': True,
    'In these constructs, the subject is being acted upon by the verb and that can muddy the waters': True,
    'He was not eating white rice': False,
    'unfortunately the story and the actors are served with a hack script': True,
    'this is not the sort of low-grade dreck that usually goes straight to video': False,
    'it will not harm anyone': False,
    'it is mildly interesting to ponder the peculiar american style of justice that plays out here': True,
    'it is a very tasteful rock and roll movie,you could put it on a coffee table anywhere ': True,
    'if you already like this sort of thing , this is that sort of thing all over again': True,
    'it does not make me feel weird': True,
    'there is not a way to effectively teach kids about the dangers of drugs': False,
    'there is no point of view': False,
    'so we are left thinking the only reason to make the movie interesting': True,
    'i am not saying that ice age does not have some fairly pretty pictures': False,
    'not only are the film is sopranos gags incredibly dated and unfunny': False,
    'although this movie makes one thing perfectly clear': True,
    'i have not been this disappointed by a movie in a long time ': False,
    'because plenty of funny movies recycle old tropes': True,
    'if you value your time and money , find an escape clause ': True,
    'the problem is not that it is all derivative': False,
    'Poems awaken the dormant soul in us': True,
    'We all come at place, where we feel dejected and disenfranchised with this materialistic world, once in our lives.': True,
    'You can prepare your own list of the TOP 100 poems.': True,
    'Her heart does not move from cold to fire easily': False

}