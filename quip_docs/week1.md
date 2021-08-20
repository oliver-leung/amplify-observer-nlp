# Week 01 (June 7-11, 2021)

## To Do’s

* Complete Embark Week 1 tasks
    * Set up Isengard
* Complete Amplify Onboarding guide
* Complete Amplify JS tutorial
* Clone `amplify-cli, -js, -docs`
* Review `yarn` commands
* Fix Discord audio
* Fix auto-call

## Questions

* Do I need to set up a Cloud Desktop?
    * Nope, other people will handle it.
* How do I fix my Cloud Desktop?
* Is it okay to use `zsh` over `bash`?
    * Yes.
* How do I set up my calendar on Android?

## Current Project Ideas

1. Create universal search function that searches issues across all Amplify repos
2. Ingest and aggregate issues across repos to determine which new features would be most helpful
    1. NLP Keyword Extraction (TF-IDF) would be useful
    2. https://www.andyfitzgeraldconsulting.com/writing/keyword-extraction-nlp/
    3. Can I use NLTK?
    4. https://docs.github.com/en/rest/reference/search#search-issues-and-pull-requests
3. Automatically categorize issues based on keywords and sentiment analysis

## Friday, June 11, 2021

* Chat w/ Stephen
    * Check out Sagemaker for AWS
    * You can deploy Amplify apps with just a simple lambda
    * Try to find things to work on that I’m already good at 
    * GH Actions can be used as a good way to “overcommunicate” with the community
    * Streamlit is a dashboarding tool
        * Used to ensure that all Amplify repos have the same setup: http://gh-he-publi-4c6e7me2gjjg-697264646.us-east-1.elb.amazonaws.com/
* Chat w/ Eddy
    * Try making own password manager
    * Train on who has worked on past related issues
    * Manuel, Francisco to talk about Amplify origin
* Platform Welcome
    * *ECBP* = Early Career Business Partner
        * People to really get to know - they have a holistic view of careers

## Thursday, June 10, 2021

* Standup meeting
    * Should talk with Mike about high-level project starting
* By Monday, should have finished tutorials
    * Going to have big meeting 
* On Monday, we’ll have a big meeting to go over my project overview
    * By then, should have finished up:

        * Official Amplify JS tutorial
        * Local Development guide for `amplify-js`
        * Isengard setup

* *Dog food:* comes from “to eat your own dog food”, to use your own product to ensure that it’s up to par
* Amplify OH
    * Interactive documentation is important
* Chat with Matt - giving overview
    * Orienting project around ML and data analysis
    * Premise - open source framework, so all libs for each language is open source
        * Lots of GH issues to deal with
            * Need to be able to reproduce steps to cause bug
            * General questions
            * Feature requests
        * Many more users than team members
            * How do we know what the community wants?
    * Each repo has an on-call enggineer dealing with issues
    * Talk with Divyesh and Eddy about Amplify Metrics
        * Has information on big issues
        * Github API
    * Ask Stephen about his work with reproduction, making it more efficient
        * New bug fix/GH issue forms
    * Is there some way to ingest issues and aggregate them?
        * Search across repos?
        * Look at React for inspiration on how issues are handled
    * Current metrics dashboard is just pulling data from GH
        * Limited by what GH API provides
        * They brought in data into a “data lake” (?) to be able to access the info better
        * Using ML to figure out which issues should be labeled as what
        * Updating Amplify Metrics with QOL stuff
* Cognito - for authorization
* *RFC* = Request For Comment
* Chat w/ Divyesh
    * Challenges
        * Requesting GH data - limited requests, and Lambda only runs for 15 mins
        * Created step function - Has different states that can wait before calling lambda and back and forth
    * Using GH App - create an app, then a bot requests data on behalf of Lambda
        * As enterprise customer
        * Getting delta of changes
        * GraphQL counts by node, not by request

## Wednesday, June 9, 2021

* AppSec: internal dept. that ensures that apps are secure
    * Patrick (AppSec contact) approved new docs page
    * Difficult to reach out to over Slack
* Builders are for building services
* T.J. is working on making Amplify Docs with Stencil, and synchronization
    * Rapid iteration is very important 

## Tuesday, June 8, 2021

* Amit is the highest manager that I would talk to
* Places to visit
    * North Bend
    * Arboretum

