// Update this variable to point to your domain.
var apigatewayendpoint = 'https://zfb2nt2z77.execute-api.us-west-2.amazonaws.com/test/issue-similarity-inference';
var loadingdiv = $('#loading');
var noresults = $('#noresults');
var resultdiv = $('#results');
var searchbox = $('input#search');
var timer = 0;

// Executes the search function 250 milliseconds after user stops typing
searchbox.keyup(function () {
  clearTimeout(timer);
  timer = setTimeout(search, 250);
});

async function search() {
  // Clear results before searching
  noresults.hide();
  resultdiv.empty();
  loadingdiv.show();
  // Get the query from the user
  let query = searchbox.val();
  // Only run a query if the string contains at least three characters
  if (query.length > 2) {
    // Make the HTTP request with the query as a parameter and wait for the JSON results
    let request = "{\"data\": \"" + query + "\"}";
    let response = await $.post(apigatewayendpoint, request, 'application/json');
    // Get the part of the JSON response that we care about
    // console.log(response);
    let results = response['Similar'];
    // console.log(results);
    if (results.length > 0) {
      loadingdiv.hide();
      resultdiv.empty();
      // Iterate through the results and write them to HTML
      resultdiv.append('<p>Found ' + results.length + ' results.</p>');
      for (var item in results) {
        let url = results[item].Url;
        let title = results[item].Title;
        // Construct the full HTML string that we want to append to the div
        resultdiv.append('<a href=\"' + url + '\">' + url + '</a> - ' + title + '<br>');
      }
    } else {
      noresults.show();
    }
  }
  loadingdiv.hide();
}

// Tiny function to catch images that fail to load and replace them
function imageError(image) {
  image.src = 'images/no-image.png';
}
