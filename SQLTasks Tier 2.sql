/* Welcome to the SQL mini project. You will carry out this project partly in
the PHPMyAdmin interface, and partly in Jupyter via a Python connection.

This is Tier 2 of the case study, which means that there'll be less guidance for you about how to setup
your local SQLite connection in PART 2 of the case study. This will make the case study more challenging for you: 
you might need to do some digging, aand revise the Working with Relational Databases in Python chapter in the previous resource.

Otherwise, the questions in the case study are exactly the same as with Tier 1. 

PART 1: PHPMyAdmin
You will complete questions 1-9 below in the PHPMyAdmin interface. 
Log in by pasting the following URL into your browser, and
using the following Username and Password:

URL: https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

In this case study, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */


/* QUESTIONS 
/* Q1: Some of the facilities charge a fee to members, but some do not.
Write a SQL query to produce a list of the names of the facilities that do. */

SELECT DISTINCT name 
FROM Facilities 
WHERE membercost > 0;


/* Q2: How many facilities do not charge a fee to members? */
/* Four (4) facilities do not charge a fee. */

SELECT COUNT(name) as count_no_fee 
FROM Facilities 
WHERE membercost = 0;


/* Q3: Write an SQL query to show a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost.
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT facid, name, membercost, monthlymaintenance 
FROM Facilities 
WHERE membercost < 0.2 * monthlymaintenance;


/* Q4: Write an SQL query to retrieve the details of facilities with ID 1 and 5.
Try writing the query without using the OR operator. */

SELECT * 
FROM Facilities 
WHERE facid IN (1,5);


/* Q5: Produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100. Return the name and monthly maintenance of the facilities
in question. */

SELECT name as facility, monthlymaintenance as monthly_maintenance, 
CASE WHEN monthlymaintenance > 100 THEN 'expensive' 
ELSE 'cheap' 
END AS facility_label 
FROM Facilities;


/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Try not to use the LIMIT clause for your solution. */

SELECT MAX(memid) as member_id,
firstname AS first_name,
surname AS last_name
FROM Members;


/* Q7: Produce a list of all members who have used a tennis court.
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

WITH f AS (
    SELECT facid,
    name 
    FROM Facilities 
    WHERE name LIKE '%Tennis Court%'
),
m AS (
    SELECT memid,
    firstname,
    surname 
    FROM members
)

SELECT 
f.name AS facility_name,
m.firstname || ' ' || m.surname AS member_name
FROM bookings AS b
INNER JOIN f
USING (facid)
INNER JOIN m
ON m.memid = b.memid
WHERE m.firstname != 'GUEST'
OR m.surname != 'GUEST'

GROUP BY m.firstname, m.surname, f.name
ORDER BY member_name;


/* Q8: Produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30. Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT 
/* Columns commented out were use for initial sanity checks*/
-- bookid AS booking, 
-- b.facid AS facility_id,
f.name AS facility_name,
-- slots AS time_slots,
-- b.memid AS member_id,
m.firstname|| ' ' || m.surname AS member_name,

-- CASE b.memid
--     WHEN 0 THEN guestcost
--     ELSE membercost 
-- END AS booking_rate,

CASE b.memid
    WHEN 0 THEN guestcost
    ELSE membercost 
END * slots AS booking_cost


FROM Bookings AS b
LEFT JOIN Facilities AS f
ON b.facid = f.facid
LEFT JOIN members AS m
ON b.memid = m.memid

WHERE date(starttime) = '2012-09-14'
AND booking_cost > 30
ORDER BY booking_cost DESC;


/* Q9: This time, produce the same result as in Q8, but using a subquery. */

SELECT
CASE
    WHEN member_name = 'GUEST GUEST' THEN 'Guest (non-member)'
    ELSE member_name
END
facility_name,
CASE 
    WHEN member_id = 0 THEN guest_rate
    ELSE member_rate 
END * time_slot AS booking_cost
FROM 
    (SELECT
    b.bookid AS booking,
    b.memid AS member_id,
    m.firstname || ' ' || m.surname AS member_name,
    b.facid AS facility_id,
    f.name AS facility_name,
    slots AS time_slot,
    membercost AS member_rate,
    guestcost AS guest_rate,
    starttime
    
    FROM bookings AS b
    LEFT JOIN facilities AS f
    ON b.facid = f.facid
    LEFT JOIN members AS m
    ON b.memid = m.memid
    WHERE date(starttime) = '2012-09-14') AS bm

WHERE booking_cost > 30 
ORDER BY booking_cost DESC;


/* PART 2: SQLite

Export the country club data from PHPMyAdmin, and connect to a local SQLite instance from Jupyter notebook 
for the following questions.  

QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

SELECT

    DISTINCT facility_name,
    ROUND(SUM(booking_cost)) AS total_revenue

FROM
    (SELECT
        b.bookid AS booking,
        b.memid AS member_id,
        m.firstname || ' ' || m.surname AS member_name,
        b.facid AS facility_id,
        f.name AS facility_name,
        slots AS time_slot,
        membercost AS member_rate,
        guestcost AS guest_rate,
        CASE 
            WHEN b.memid = 0 THEN guestcost
            ELSE membercost  
        END * slots AS booking_cost,
        starttime

    FROM Bookings AS b
        LEFT JOIN Facilities AS f
        ON b.facid = f.facid
        LEFT JOIN Members AS m
        ON b.memid = m.memid

    ) AS bm

GROUP BY facility_name
HAVING total_revenue < 1000
ORDER BY total_revenue;

/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */

SELECT
    m1.surname ||', ' || m1.firstname AS member_name,
    COALESCE(m2.surname ||', ' || m2.firstname, 'No recommender') AS recommender_name
FROM members AS m1
    LEFT JOIN members AS m2
    ON m1.recommendedby = m2.memid
WHERE m1.memid NOT IN (0)
ORDER BY member_name;

/* Q12: Find the facilities with their usage by member, but not guests */

WITH
    member_time_CTE
    AS
    (
        SELECT
            memid,
            facid,
            slots,
            SUM(
            CASE WHEN memid != 0 THEN slots END 
            ) AS member_time
        FROM bookings
        GROUP BY facid
    ),
    guest_time_CTE
    AS
    (
        SELECT
            memid,
            facid,
            slots,
            SUM(
        CASE WHEN memid = 0 THEN slots END 
        ) AS guest_time
        FROM bookings
        GROUP BY facid
    )

SELECT
    f.name AS facility_name,
    SUM(b.slots) AS total_time_slots,
    m.member_time,
    -- g.guest_time,
    ROUND(CAST(m.member_time AS float)/SUM(b.slots) , 3)  AS member_pct_usage
-- ROUND(CAST(g.guest_time AS float)/SUM(b.slots) , 3)  AS guest_pct_usage
FROM bookings AS b
    LEFT JOIN member_time_CTE AS m
    ON b.facid = m.facid
    LEFT JOIN guest_time_CTE AS g -- check the guest time to see if it complements the member's pct
    ON b.facid = g.facid
    LEFT JOIN facilities AS f
    ON b.facid = f.facid
GROUP BY b.facid
ORDER BY member_pct_usage;

/* Q13: Find the facilities usage by month, but not guests */

WITH member_time_CTE AS (
    SELECT
        strftime('%m', date(starttime)) AS month_,
        memid,
        facid,
        slots,
        SUM(SUM(
            CASE WHEN memid != 0 THEN slots 
            END 
        )) OVER(PARTITION BY facid , strftime('%m', date(starttime)) ) AS member_time
    FROM bookings
    GROUP BY month_, facid
),
guest_time_CTE AS (
    SELECT
        strftime('%m', date(starttime)) AS month_,
        memid,
        facid,
        slots,
        SUM(SUM(
            CASE WHEN memid = 0 THEN slots 
            END 
        )) OVER(PARTITION BY facid ,strftime('%m', date(starttime)) ) AS guest_time
    FROM bookings
    GROUP BY month_, facid
    )

SELECT
    strftime('%m', date(b.starttime)) AS month_no,
    f.name AS facility_name,
    SUM(SUM(
        b.slots
    )) OVER(PARTITION BY b.facid, strftime('%m', date(b.starttime)) ) AS total_time_slots,
    m.member_time,
    -- g.guest_time,
    ROUND(CAST(m.member_time AS float)/SUM(b.slots) , 3)  AS member_pct_usage
-- ROUND(CAST(g.guest_time AS float)/SUM(b.slots) , 3)  AS guest_pct_usage
FROM bookings AS b
    LEFT JOIN member_time_CTE AS m
    ON b.facid = m.facid AND month_no = m.month_
    LEFT JOIN guest_time_CTE AS g -- check the guest time to see if it complements the member's pct
    ON b.facid = g.facid AND month_no = g.month_
    LEFT JOIN facilities AS f
    ON b.facid = f.facid
GROUP BY  b.facid, month_no
ORDER BY month_no, member_pct_usage;

