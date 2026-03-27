local points = {
    -- {x, y, z, hold_time}
    { 0.0,  0.0, 1.5, 1.0},
    { 1.0,  0.0, 2.0, 5.0},
    { 1.0,  1.0, 2.0, 1.0},
    { 0.0,  1.0, 1.8, 1.0},
    {-1.0,  1.0, 2.2, 5.0},
    {-1.0,  0.0, 1.7, 1.0},
    { 0.0, -1.0, 1.5, 1.0}
}

local currentPoint = 1

local function flyToNextPoint()
    if currentPoint > #points then
        ap.push(Ev.MCE_LANDING)
        return
    end

    local p = points[currentPoint]
    local x = p[1]
    local y = p[2]
    local z = p[3]

    ap.goToLocalPoint(x, y, z)
end

function callback(event)
    if event == Ev.TAKEOFF_COMPLETE then
        flyToNextPoint()
    end

    if event == Ev.POINT_REACHED then
        local holdTime = points[currentPoint][4]

        Timer.callLater(holdTime, function()
            currentPoint = currentPoint + 1
            flyToNextPoint()
        end)
    end
end

ap.push(Ev.MCE_PREFLIGHT)
Timer.callLater(1, function()
    ap.push(Ev.MCE_TAKEOFF)
end)