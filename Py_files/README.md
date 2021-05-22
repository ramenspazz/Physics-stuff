# Dalton Tinoco, Interview questions

- Question 1:  

>```
SELECT * FROM class_tbl
WHERE SUBSTR(class_tbl.INSTRUCTOR,1,1) = 'T';```

- Question 2:

>```
SELECT * FROM class_tbl
WHERE class_tbl.ENROLLED_COUNT > 5 AND class_tbl.CLASS_CAPACITY = 100;```

- Question 3:

>```
SELECT DISTINCT CLASS_SUBJECT FROM class_tbl;
```

- Question 4:

>```
SELECT CLASS_SUBJECT, COUNT(*) FROM class_tbl GROUP BY CLASS_SUBJECT;```

- Question 5:

>```
SELECT INSTRUCTOR, COUNT(*) FROM class_tbl GROUP BY INSTRUCTOR;```

- Question 6:

>```
SELECT INSTRUCTOR, TERM_OFFERED, CLASS_SUBJECT, COUNT(CLASS_SECTION)
FROM class_tbl GROUP BY INSTRUCTOR, TERM_OFFERED, CLASS_SUBJECT;```

- Question 7:

>```
SELECT *
FROM class_tbl
WHERE class_tbl.class_capacity - class_tbl.enrolled_count > 49;```

- Question 8:

>```
SELECT *
FROM class_tbl
WHERE class_tbl.CLASS_SUBJECT = 'BIO' AND (class_tbl.INSTRUCTOR = 'ROMERO' OR class_tbl.INSTRUCTOR = 'GONZALEZ');```

- Question 9:

>```
SELECT DISTINCT CONCAT(SUBSTR(INSTRUCTOR,1,1),LOWER(SUBSTR(INSTRUCTOR,2,LENGTH(INSTRUCTOR))))
FROM class_tbl;```
