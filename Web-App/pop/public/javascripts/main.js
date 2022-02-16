function addDays(date, days) {
    var result = new Date(date);
    result.setDate(result.getDate() + days);
    return result;
}

DEFAULT_POI_ID = 24
selected_poi = DEFAULT_POI_ID
addr = "http://127.0.0.1:5000"
fidelity = ""
current_data = {}
current_date = ""
wday = 0
day_shift = 0
week_shift = 0
last_update = document.getElementById("last_update")
current_day = document.getElementById("current_day")
chart = document.getElementById("chart")


function current(){
    document.getElementById("back_button").disabled = true
    rq = { "WEEKDAY": (wday%7), "WEEK_SHIFT": Math.floor(day_shift/7), "POI_ID": selected_poi}
    $.ajax({
        url: addr+'/current',
        type: 'POST',
        dataType: "json",
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(rq),
        responseType:'json',
        crossDomain: true,
        success: function(response) {
            current_data = $.parseJSON(response)
            SAMPLING_TIME = Number.parseInt(current_data["SAMPLING_TIME"])
            wday = Number.parseInt(current_data["WEEKDAY"])
            count = Object.keys(current_data).length
            count -= 4
            hour = Number.parseInt((count * 3600) / SAMPLING_TIME)
            time = ""
            if(hour<10) time = "0"+hour+":00"
            else time = hour+":00"
            current_date = current_data["DATE"]
            x = current_date.split("/")
            current_date = (Number.parseInt(x[2])+2000) + "-" + x[1] + "-" + x[0]
            current_date = Date.parse(current_date)
            last_update.innerHTML=current_data["DATE"] + " - " + time
            postPrediction({ "WEEKDAY": wday, "WEEK_SHIFT": 0 ,"POI_ID": selected_poi})
        },
        error: function (xhr, status) {
            alert(status);
        }
    });
}

function postPrediction(request){
    today = (day_shift == 0 && week_shift == 0)
    if(today){
        document.getElementById("back_button").disabled = true
    } else {
        document.getElementById("back_button").disabled = false
    }
    $.ajax({
        url: addr+'/predict',
        type: 'POST',
        dataType: "json",
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(request),
        responseType:'json',
        crossDomain: true,
        success: function(response) {
            if(today){
                predicted_today = $.parseJSON(response)
                end = Number.parseInt((3600) / SAMPLING_TIME) * 23
                x = time.split(":")
                idx = Number.parseInt(x[0])
                for(i=idx+1;i<=end;i++){
                    r = ""
                    if(i<10) r = "0"+i+":00"
                    else r = i+":00"
                    current_data[r] = predicted_today[r]
                }
                colors = []
                c2 = Object.keys(current_data).length
                c2 -= 3
                for(i=0;i<c2;i++){
                    if(i<idx+1) colors.push("#f25a5a")
                    else colors.push("#4f73df")
                }
                updateChart(colors)
            } else {
                current_data = $.parseJSON(response)
                updateChart()
            }
        },
        error: function (xhr, status) {
            alert(status);
        }
    });
}

$("#today_button").click(function(e) {
    current()
    day_shift = 0
    current_date = addDays(current_date,-wday)
    wday = 0
    current_day.innerHTML = "TODAY"
    document.getElementById("back_button").disabled = true
});

$("#back_button").click(function(e) {
    e.preventDefault()
    document.getElementById("next_button").disabled = false
    if(day_shift==1){
        current()
        day_shift = 0
        wday -= 1
        current_date = addDays(current_date,-1)
        current_day.innerHTML = "TODAY"
        document.getElementById("back_button").disabled = true
    } else {
        wday -= 1
        day_shift -= 1
        current_date = addDays(current_date,-1)
        current_day.innerHTML = current_date.getDate() + "/" + (current_date.getMonth() + 1) + "/" + current_date.getFullYear()
        postPrediction({ "WEEKDAY": (wday%7), "WEEK_SHIFT": Math.floor(day_shift/7), "POI_ID": selected_poi})
    }
});

$("#next_button").click(function(e) {
    if(day_shift==30){
        document.getElementById("next_button").disabled = true
    }
    wday += 1
    day_shift += 1
    current_date = addDays(current_date,1)
    current_day.innerHTML = current_date.getDate() + "/" + (current_date.getMonth() + 1) + "/" + current_date.getFullYear()
    document.getElementById("back_button").disabled = false
    postPrediction({ "WEEKDAY": (wday%7), "WEEK_SHIFT": Math.floor(day_shift/7), "POI_ID": selected_poi})
});

current()

var lx = [];
for(i=0;i<24;i++){
    h = i;
    if(h<10) h = "0" + i
    hour = h+":00"
    lx.push(hour)
}

var ctx = document.getElementById("chart_data");
var barChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: lx,
        datasets: [{
            backgroundColor: [""],
            label: 'Prediction',
            data: [""]
        }]
    },
    options: {
        legend: {
            display: false
        },
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero: true,
                    min: 0,
                    max: 1,
                    stepSize: 0.25,
                }
            }]
        }
    }
});

function updateChart(colors){
    var cx = [];
    if(typeof colors !== 'undefined'){
        cx = colors
    }
    var dat = [];
    for(i=0;i<24;i++){
        h = i;
        if(h<10) h = "0" + i
        hour = h+":00"
        if(typeof colors == 'undefined') cx.push("#4f73df")
        x = Number.parseFloat(current_data[hour])
        dat.push(x)
    }
    updateBarGraph(barChart,'Prediction', cx, dat);
}

/*Function to update the bar chart*/
function updateBarGraph(chart, label, color, data) {
    chart.data.datasets.pop();
    chart.data.datasets.push({
        label: label,
        backgroundColor: color,
        data: data
    });
    chart.update();
    week_shift = Math.floor(day_shift/7)
    if(week_shift < 1){
        document.getElementById("fidelity_field").innerHTML = "High fidelity"
    } else if(week_shift >= 1 && week_shift < 2){
        document.getElementById("fidelity_field").innerHTML = "Medium fidelity"
    } else {
        document.getElementById("fidelity_field").innerHTML = "Low fidelity"
    }
}

