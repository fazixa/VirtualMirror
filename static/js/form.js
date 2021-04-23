
function myFunction(r, g, b) {
    // alert("I am an alert box!");
    $.ajax({
        data: {
            r: r,
            g: g,
            b: b
        },
        type: 'POST',
        url: '/process'
    })
}

function camrelease() {
    $.ajax({
        data: {
        },
        type: 'POST',
        url: '/caprelease'
    })
}

function opencam() {
    $.ajax({
        data: {
        },
        type: 'POST',
        url: '/opencam'
    })
}