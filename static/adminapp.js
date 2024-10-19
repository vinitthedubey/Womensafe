var config = {
    cUrl: 'https://api.countrystatecity.in/v1/countries',
    ckey: 'NHhvOEcyWk50N2Vna3VFTE00bFp3MjFKR0ZEOUhkZlg4RTk1MlJlaA=='
};

var countrySelect = document.querySelector('.country'),
    stateSelect = document.querySelector('.state'),
    citySelect = document.querySelector('.city');

// Load countries when the page is ready
document.addEventListener('DOMContentLoaded', loadCountries);

// Fetch Countries
async function loadCountries() {
    try {
        let response = await fetch(config.cUrl, {
            headers: { "X-CSCAPI-KEY": config.ckey }
        });
        let data = await response.json();

        // Populate country dropdown
        data.forEach(country => {
            const option = document.createElement('option');
            option.value = country.iso2;
            option.textContent = country.name;  // Send the text
            countrySelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading countries:', error);
    }

    stateSelect.disabled = true;
    citySelect.disabled = true;
}

// Fetch States based on selected country
async function loadStates() {
    try {
        const selectedCountryCode = countrySelect.value;
        stateSelect.innerHTML = '<option value="">Select State</option>';
        citySelect.innerHTML = '<option value="">Select City</option>';

        let response = await fetch(`${config.cUrl}/${selectedCountryCode}/states`, {
            headers: { "X-CSCAPI-KEY": config.ckey }
        });
        let data = await response.json();

        stateSelect.disabled = false;
        data.forEach(state => {
            const option = document.createElement('option');
            option.value = state.iso2;
            option.textContent = state.name;  // Send the text
            stateSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading states:', error);
    }
}

// Fetch Cities based on selected state
async function loadCities() {
    try {
        const selectedCountryCode = countrySelect.value;
        const selectedStateCode = stateSelect.value;
        citySelect.innerHTML = '<option value="">Select City</option>';

        let response = await fetch(`${config.cUrl}/${selectedCountryCode}/states/${selectedStateCode}/cities`, {
            headers: { "X-CSCAPI-KEY": config.ckey }
        });
        let data = await response.json();

        citySelect.disabled = false;
        data.forEach(city => {
            const option = document.createElement('option');
            option.value = city.iso2;
            option.textContent = city.name;  // Send the text
            citySelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading cities:', error);
    }
}

// Submit data function with async-await
async function submitData() {
    try {
        const countryText = countrySelect.options[countrySelect.selectedIndex].text;  // Get selected country text
        const stateText = stateSelect.options[stateSelect.selectedIndex].text;  // Get selected state text
        const cityText = citySelect.options[citySelect.selectedIndex].text;  // Get selected city text

        console.log(`Selected Country: ${countryText}, State: ${stateText}, City: ${cityText}`);  // Debugging

        // Send data to the backend via POST
        let response = await fetch('/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ country: countryText, state: stateText, city: cityText })
        });

        let data = await response.json();

        if (data.error) {
            alert(data.error);
        } else {
            // Plotly data visualization
            Plotly.newPlot('pie-chart', JSON.parse(data.pie));
            Plotly.newPlot('bar-incidents', JSON.parse(data.bar_incidents));
            Plotly.newPlot('bar-reports', JSON.parse(data.bar_reports));

            const searchQuery = `recent women related crimes in ${cityText}`;
            document.getElementById('google-search-results').innerHTML = `<a href="https://www.google.com/search?q=${searchQuery}" target="_blank">${searchQuery}</a>`;
        }
    } catch (error) {
        console.error('Error submitting data:', error);
    }
}

// Function for updating Safe/Unsafe
async function updateSafe(safe) {
    try {
        const countryText = countrySelect.options[countrySelect.selectedIndex].text;
        const stateText = stateSelect.options[stateSelect.selectedIndex].text;
        const cityText = citySelect.options[citySelect.selectedIndex].text;

        let response = await fetch('/update_safe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ country: countryText, state: stateText, city: cityText, safe: safe })
        });

        let data = await response.json();
        console.log(data);

        document.body.style.backgroundColor = safe ? 'green' : 'red';
    } catch (error) {
        console.error('Error updating safety status:', error);
    }
}

countrySelect.addEventListener('change', loadStates);
stateSelect.addEventListener('change', loadCities);
