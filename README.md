- Project overview
    Shoply, the AI shopping assistant, utilising the Haiku 4.5 model to help users shop for their desired language as if they are talking to a sales assistant in a physical store.

    Uses the anthropic python package to link the api and a scoring system to see how relevant items are to the users needs. Item that the user will be looking for is shown on the left as a sample item the user is looking for, which is data that is hidden from the AI and the algorithm. The user will then interact with the AI shopping assistant to eventually arrive at the desired target item.

    There is also a simulation mode where the user inputs are replaced by another AI that knows the target item and describes it in natural language and the shopping assistant will have to figure out what the right item is. 

- Setup and installation instructions
    1. Generate an api key for haiku 4.5
    2. Download the and unzip the project package
    3. Launch the executable called server.py
    4. On the top left, input your api key
    The shopping assistant will be ready for use.

- Steps to reproduce your results
    1. Select any case in the drop down menu to select a random item
    2. View what the item is and its descriptors on the left panel which AI cannot see
    3. In natural language ask the shopping assistant for what you are looking for
    4. The shopping assistant will respond with its conclusions and follow up questions so do respond accordingly.
    5. After a number of turns it will show that your item is in its shortlist of the top 10 items, completing its task.
    `
- A brief reflection on your solution's limitations and what you would improve given more time
    - It is limited by the user having their own api key and tokens in anthropic.
    - Would like to improve the UI because it currently looks very minimal and some of the information there is more for development use rather than consumer uses
    - Currently it does not save between launches of the program as therer is no persisting memory