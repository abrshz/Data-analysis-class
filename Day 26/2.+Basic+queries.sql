USE Sample_Schema;

SELECT * FROM courses_data;

SELECT Credits FROM courses_data;

SELECT * FROM student_information;

SELECT DISTINCT Department FROM student_information;

SELECT GPA AS "Grade Point Average"
FROM student_information;

SELECT  c.Credits, e.Grade, s.GPA
FROM courses_data AS c, enrollments_data AS e, student_information AS s;

SELECT Gender, Department, GPA
FROM student_information
WHERE Age = 20;
