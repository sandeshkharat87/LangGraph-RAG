class Cstm_Greetings():
    def __init__(self,word):
        self.word = word


    def __call__(self,name):
        return f"Hello {name}, yo: {self.word}" 
       


yo = Cstm_Greetings(word="damnnn")


print(yo(name="sandesh"))
