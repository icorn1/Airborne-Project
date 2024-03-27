package com.arbrn.urcap.demonstration.impl;

import com.arbrn.urcap.demonstration.communicator.ScriptCommand;
import com.arbrn.urcap.demonstration.communicator.ScriptExporter;
import com.arbrn.urcap.demonstration.communicator.ScriptSender;
import com.ur.urcap.api.contribution.InstallationNodeContribution;
import com.ur.urcap.api.domain.script.ScriptWriter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;

public class TestInstallationNodeContribution implements InstallationNodeContribution {

	private final TestInstallationNodeView view;
	
	// Instance of the ScriptSender
	// Used to send a URScript for execution
	private final ScriptSender sender;
	
	// Instance of ScriptExporter
	// Used to extract values from URScript
	private final ScriptExporter exporter;
	
	public TestInstallationNodeContribution(TestInstallationNodeView view) {
		this.view = view;
		
		// Initialize Sender and Exporter here... 
		this.sender = new ScriptSender();
		this.exporter = new ScriptExporter();
	}
	
	/**
	 * This command is invoked by clicking the SEND button in the View 
	 * Sends a popup command to URControl
	 */

	public void sendScriptTest(String action, String laminateId) {
		// Create a new ScriptCommand called "testSend"
		ScriptCommand sendTestCommand = new ScriptCommand();
		
		try {
			// Read the contents of the file "full_code.script" from the resources
			InputStreamReader inputStreamReader = new InputStreamReader(Objects.requireNonNull(this.getClass().getResourceAsStream("/com/arbrn/urcap/demonstration/full_code.script")));
			BufferedReader reader = new BufferedReader(inputStreamReader);
			String line;
			// Append each line of the file to the ScriptCommand
			while ((line = reader.readLine()) != null) {
				sendTestCommand.appendLine(line);
			}
			reader.close();
			
			sendTestCommand.appendLine("Initialise()\n");

			if (action == "calibration("){
				sendTestCommand.appendLine(action + ")\n");
			}
			else {
				sendTestCommand.appendLine(action + laminateId + ")\n");
			}

			sendTestCommand.appendLine("De_initialise()\n");

			System.out.println(sendTestCommand.toString());
			
		} catch (NullPointerException e) {
			e.printStackTrace();
			// Handle resource not found exception here
		} catch (IOException e) {
			e.printStackTrace();
			// Handle file reading or writing errors here
		}
		
		// Use the ScriptSender to send the command for immediate execution
		sender.sendScriptCommand(sendTestCommand);
	}

	@Override
	public void openView() {
	}

	@Override
	public void closeView() {
	}

	@Override
	public void generateScript(ScriptWriter writer) {
	}
}
