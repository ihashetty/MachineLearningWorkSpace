
from flask import Flask,jsonify

class Food:
    def __init__(self,id,food_name,price):
        self.id=id
        self.food_name= food_name
        self.price=price  
    
    def to_dict(self):
        return {"id":self.id,"name":self.food_name,"price":self.price}    

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello Flask!"

@app.route("/test")
def test():
    return "Test Method!"

@app.route("/nums")
def nums():
    nums={"num1":20,"num2":45,"num3":11}
    return nums

@app.route("/displayItems")
def display_food_items():
    food_items=[Food(101,"burger",100),Food(102,"pizza",200)]
    return food_items

@app.route("/displayFood")
def display_food():
    return jsonify(Food(101,"burger",100).to_dict())

if __name__ == "__main__":
    app.run(debug=True)
