# Wissenschaftliches Rechnen - Tests für die HA's

## big big recommendation:

install this VSCode extension: 
- [Python Test Explorer for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=LittleFoxTeam.vscode-python-test-adapter)

short setup needed:
1. click on the new testing field on the sidebar
2. select 'configure python tests'
3. select 'unittest'
4. select '. Root directory'
5. select '\*test\*.py'

go into the tests.py file and enjoy being able to click on the individual test or test them on the new field on the sidebar


## how to run the tests
#
1. Kopiere die Datei `tests.py` des aktuellen Aufgabenblattes in dein Projekt.
2. Führe anschließend im Terminal diesen Befehl aus (je nach Python-Version):


-   ```shell
    python3 tests.py -v Tests.test_all
    ```
    _oder_

-   ```shell
    python tests.py -v Tests.test_all
    ```
    
    Wenn du doch einzeln testen möchtest, dann wie im Aufgabenblatt beschrieben:
-   ```shell
    python3 tests.py -v Tests.Funktionsname
    ```
#
### made by me :D

### auf ein gutes Semester!