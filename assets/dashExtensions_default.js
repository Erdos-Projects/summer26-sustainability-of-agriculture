window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature) {
            return {
                fillColor: feature.properties.color
            };
        }
    }
});