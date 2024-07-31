import pyttsx3

# if __name__ == '__main__':, it means that the code block inside the if statement will only be executed if the 
# script is being run directly as the main program.

if __name__ == '__main__':
    print("Created By Abdurrehman")
    engine = pyttsx3.init()
    while True:
        x = input("Enter What you want me to Pronounce: ")
        if x.lower() == "q":
            engine.say("Bye Bye friend")
            engine.runAndWait()
            break
        engine.say(x)
        engine.runAndWait()
